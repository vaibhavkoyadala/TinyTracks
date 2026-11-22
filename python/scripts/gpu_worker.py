"""The tinygrad worker: all tinygrad work runs on the process's main thread.

Why the main thread:
1. tinygrad's BEAM search installs signal handlers, and Python only allows that on the main thread.
2. Metal objects must not outlive the thread that created them. Model work on short-lived
   request threads segfaults when those threads exit. The main thread never exits.
3. tinygrad keeps global state (JIT caches, BEAM settings), so one thread must do all of it.

The HTTP servers run on background threads and hand work to this loop through call()/stream().
Call run_forever() from the main thread; it returns after stop().
"""
from __future__ import annotations

import concurrent.futures
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any


class GpuWorker:
    def __init__(self) -> None:
        self._tasks: queue.Queue[Callable[[], None]] = queue.Queue()
        self._stopping = threading.Event()
        self._owner: threading.Thread | None = None

    def run_forever(self) -> None:
        """Process tasks until stop(). Must be called from the main thread."""
        self._owner = threading.current_thread()
        while not self._stopping.is_set():
            try:
                task = self._tasks.get(timeout=0.2)  # short timeout so signal handlers can run
            except queue.Empty:
                continue
            task()

    def stop(self) -> None:
        self._stopping.set()

    def in_worker(self) -> bool:
        return self._owner is not None and threading.current_thread() is self._owner

    def call(self, fn: Callable[[], Any]) -> Any:
        """Run fn on the worker thread and return its result (or re-raise its error)."""
        if self.in_worker():
            return fn()
        future: concurrent.futures.Future = concurrent.futures.Future()

        def task() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(fn())
            except BaseException as exc:  # propagate everything to the caller
                future.set_exception(exc)

        self._tasks.put(task)
        return future.result()

    def stream(self, start: Callable[[], Iterator[Any]]) -> Iterator[Any]:
        """Run a generator on the worker, one step per task, yielding each non-None value as it arrives.

        Each step is queued behind whatever else is waiting, so a long generation on one model
        does not block the others. Only the step itself runs on the worker thread.
        If the consumer stops early, the generator is closed on the worker at its next step.
        """
        out: queue.Queue = queue.Queue()
        done = object()
        cancel = threading.Event()
        holder: list[Iterator[Any]] = []

        def step() -> None:
            try:
                if not holder:
                    holder.append(start())
                if cancel.is_set():
                    holder[0].close()
                    out.put(done)
                    return
                item = next(holder[0])
            except StopIteration:
                out.put(done)
                return
            except BaseException as exc:
                out.put(_Failure(exc))
                return
            if item is not None:
                out.put(item)
            self._tasks.put(step)  # re-queue at the back: other work runs before our next token

        self._tasks.put(step)
        try:
            while True:
                item = out.get()
                if item is done:
                    return
                if isinstance(item, _Failure):
                    raise item.exc
                yield item
        finally:
            cancel.set()


class _Failure:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc
