"""Entry point for the bundled Python backend.

Electron launches this as:
    python worker.py --host 127.0.0.1 --port <free port>
with TINYTRACKS_TOKEN set in the environment.

Threads:
  main thread      runs the tinygrad worker loop (all model work; BEAM needs the main thread)
  thread           the internal API for the Electron window (--port)
  thread           the OpenAI-compatible API for live models (TINYTRACKS_OPENAI_PORT, default 8080)
"""
import argparse
import ipaddress
import os
import signal
import sys
import threading

# Compile kernels in-process. tinygrad's worker pool forks, and forking after Metal starts crashes.
os.environ.setdefault("PARALLEL", "0")

import uvicorn  # noqa: E402

import api  # noqa: E402  (builds the managers; tinygrad is imported here)
from openai_api import OpenAIServer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyTracks tinygrad worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    # Only ever bind to loopback; this server is not meant for the network.
    if not ipaddress.ip_address(args.host).is_loopback:
        sys.exit("Refusing to bind to a non-loopback address")

    internal = uvicorn.Server(uvicorn.Config(api.app, host=args.host, port=args.port, log_level="info"))
    internal_thread = threading.Thread(target=internal.run, name="internal-api", daemon=True)
    internal_thread.start()

    openai_server = OpenAIServer(api.serving, api.OPENAI_HOST, api.OPENAI_PORT)
    try:
        openai_server.start()
        api.openai_error = None
        print(f"[tinytracks] OpenAI-compatible API on http://{api.OPENAI_HOST}:{api.OPENAI_PORT}/v1", flush=True)
    except RuntimeError as exc:
        api.openai_error = str(exc)
        print(f"[tinytracks] {exc}", flush=True)

    def shutdown(*_):
        api.gpu.stop()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        api.gpu.run_forever()  # returns after SIGTERM/SIGINT
    finally:
        internal.should_exit = True
        openai_server.stop()
        internal_thread.join(timeout=5)
        api.manager.shutdown()
        api.benchmarks.stop()
        sys.stdout.flush()
        sys.stderr.flush()
        # Skip interpreter teardown: releasing Metal objects at exit can segfault.
        os._exit(0)


if __name__ == "__main__":
    main()
