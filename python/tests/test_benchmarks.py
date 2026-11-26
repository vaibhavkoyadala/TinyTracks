"""Auto-optimize must run tinygrad's BEAM search for every width it measures.

Run from the project root with the bundled interpreter:
    python/venv/bin/python -m unittest discover -s python/tests -v

tinygrad, the model loader and the GPU are replaced with fakes, so this runs without
weights or a GPU. What is checked is which BEAM value each measurement runs under.
"""
from __future__ import annotations

import sys
import threading
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import benchmarks  # noqa: E402
from store import Store  # noqa: E402

MODEL_ID = "smollm2-135m-instruct"


class RecordingContext:
    """Stands in for tinygrad's Context and records the BEAM value each block ran under."""

    entered: list[int] = []

    def __init__(self, **kwargs):
        self.beam = kwargs.get("BEAM", 0)

    def __enter__(self):
        RecordingContext.entered.append(self.beam)
        return self

    def __exit__(self, *exc):
        return False


class FakeModel:
    _cached_tokens: list = []

    def generate(self, ids, temperature=0.0):
        for i in range(1000):
            yield i


class FakeLoadedModel:
    def __init__(self, *args, **kwargs):
        self.model = FakeModel()

    def prompt_ids(self, messages):
        return [1, 2, 3]

    def close(self):
        pass


class FakeServing:
    def check_servable(self, model_id):
        return None

    def model_dir(self, model_id):
        return Path("/nonexistent")

    def mark_busy(self, model_id, busy):
        return None


class InlineGpu:
    """Runs the measurement on the calling thread instead of the GPU worker."""

    def call(self, fn):
        return fn()


class AutoOptimizeBeamTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._tmp.name) / "test.db")
        self.patches = [
            (benchmarks, "Context", RecordingContext),
            (benchmarks, "LoadedModel", FakeLoadedModel),
            (benchmarks, "_beam_rows", staticmethod(lambda: {})),
        ]
        self._saved = [(obj, name, getattr(obj, name)) for obj, name, _ in self.patches]
        for obj, name, value in self.patches:
            setattr(obj, name, value)
        RecordingContext.entered = []

        self.manager = benchmarks.BenchmarkManager(self.store, FakeServing(), InlineGpu())
        # Keep the search loop synchronous: same bookkeeping as _start, but _run on this thread.
        def start_inline(job_id):
            resume = threading.Event()
            resume.set()
            self.manager._resume[job_id] = resume
            self.manager._cancel[job_id] = threading.Event()
            self.manager._run(job_id)

        self.manager._start = start_inline

    def tearDown(self):
        for obj, name, value in self._saved:
            setattr(obj, name, value)
        self._tmp.cleanup()

    def test_auto_measures_every_width_under_beam_search(self):
        self.manager.create(MODEL_ID, widths=[], gen_tokens=8, prompt=None, auto=True)

        # Width 0 is the baseline (BEAM=0). Every later width must run a real search (BEAM=width).
        self.assertEqual(RecordingContext.entered[0], 0)
        searched = [b for b in RecordingContext.entered if b]
        self.assertEqual(searched, list(range(1, benchmarks.AUTO_MAX_WIDTH + 1)))

    def test_auto_stops_at_auto_max_width(self):
        self.manager.create(MODEL_ID, widths=[], gen_tokens=8, prompt=None, auto=True)
        job = self.manager.list()[0]
        widths = [r["beam"] for r in self.store.list_results(job["id"])]
        self.assertEqual(widths, list(range(0, benchmarks.AUTO_MAX_WIDTH + 1)))
        self.assertEqual(self.store.get_job(job["id"])["status"], "done")

    def test_manual_widths_still_run_beam_search_for_each_width(self):
        self.manager.create(MODEL_ID, widths=[0, 2, 4], gen_tokens=8, prompt=None, auto=False)
        self.assertEqual(RecordingContext.entered, [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
