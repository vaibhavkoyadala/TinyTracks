# TinyTracks

Electron desktop app. The UI (`renderer/`) talks to a bundled Python server (`python/scripts/`) that runs tinygrad.

```
TinyTracks/
├── electron/          # main.js (spawns Python, IPC) + preload.js (exposes window.backend)
├── renderer/          # index.html, app.js, styles.css — the web page
├── python/
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── worker.py      # entry point: starts the HTTP server on a loopback port
│   │   ├── api.py         # FastAPI routes (tinygrad lives here)
│   │   ├── catalog.py     # the models that can be downloaded (id, URL, size, SHA-256)
│   │   ├── store.py       # SQLite database: model rows and their state
│   │   └── downloads.py   # resumable, verified downloads with progress tracking
│   ├── data/              # created at runtime: tinytracks.db + weights/<model>/ (git-ignored)
│   ├── bin/ lib/ include/ share/  # python-build-standalone 3.12 runtime (unpacked)
│   └── venv/                      # deps: tinygrad, fastapi, uvicorn, pydantic
└── package.json       # extraResources copies python/ into the app bundle
```

## How it works

1. `electron/main.js` picks a free loopback port and generates a random token.
2. It starts `python/scripts/worker.py --port <port>` with `TINYTRACKS_TOKEN` in the environment.
3. The renderer calls `window.backend.request(method, '/api/...', body)`. The preload forwards this over IPC to the main process, which adds the token and calls the Python server. The renderer never talks to the server directly and never sees the token.

## Models

The Models tab lists the catalog, starts and pauses downloads, and deletes weights.

- **Storage.** State lives in `data/tinytracks.db`, a single SQLite file. It needs no server, and Python ships with it. In a packaged app the data goes to the per-user app data folder (`TINYTRACKS_DATA_DIR`, set by `electron/main.js`). Weights are saved as `data/weights/<model-id>/model.safetensors`.
- **Downloads.** Bytes go to a `.part` file. The file is renamed into place only after its size and SHA-256 match the catalog. Interrupted transfers resume with an HTTP `Range` request, and network errors retry with backoff. Downloads that were running when the app closed resume on the next launch. A disk-space check runs before each start.
- **Security.** Clients can only pick a catalog id. They never supply a URL or a path.

### Internal API (token-gated, used by the window)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/models` | Catalog with download state, serving state and progress |
| POST | `/api/models/{id}/download` | Start or resume a download (202) |
| POST | `/api/models/{id}/download/pause` | Pause a download and keep the partial files |
| DELETE | `/api/models/{id}` | Delete weights (409 while loaded or downloading) |
| POST | `/api/models/{id}/load` | Load weights into memory (paused) |
| POST | `/api/models/{id}/play` | Load if needed, then go live |
| POST | `/api/models/{id}/pause` | Stop serving; stays loaded |
| POST | `/api/models/{id}/eject` | Unload from memory (409 during a benchmark) |
| GET | `/api/serving` | Dashboard: base URL, key, live models with endpoint and curl |
| POST | `/api/serving/rotate-key` | New API key (breaks existing clients) |
| GET/POST | `/api/benchmarks` | List or start a beam-search job |
| GET | `/api/benchmarks/{job}` | One job with its results (`best_beam` marked) |
| POST | `/api/benchmarks/{job}/pause` \| `resume` \| `cancel` | Control a job |
| POST | `/api/benchmarks/{job}/auto` | Turn auto-optimize on or off |
| GET | `/api/kernels?model_id=` | Kernel options that BEAM search found |

Download statuses: `not_downloaded`, `downloading`, `paused`, `verifying`, `ready`, `failed`.
Serving states: `downloaded` (on disk), `loaded` (in memory, paused), `live` (serving).

### OpenAI-compatible API for live models

Live models are served at `http://127.0.0.1:8080/v1` (set `TINYTRACKS_OPENAI_PORT` to change it).
The model id goes in the request body, as in the photon-tap gateway:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer <key from the Dashboard>" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-0.5b-instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```

- `GET /v1/models` lists only live models. `POST /v1/chat/completions` supports `stream`, `max_tokens`,
  `max_completion_tokens`, `temperature` and `stream_options.include_usage`.
- Auth: `Authorization: Bearer` or `x-api-key`. The key is stored in the database and stays the same across
  restarts until you rotate it.
- Errors use the OpenAI shape, `{"error": {"message", "type", "code"}}`: 401 bad key, 404 unknown model,
  503 model not live, 400 bad request or context too long.
- Not implemented: `top_p`, tools, and stored responses.

### What can be served

Only Llama and Qwen2 checkpoints can be served live (SmolLM2-135M, Qwen2.5-0.5B). The other catalog
models show "not servable yet" and their Play button stays disabled. Weights must fit in memory, because
they are copied onto the GPU when loaded.

### Beam search benchmarks

A job runs one measurement per beam width. `0` is the baseline with no kernel search. Each width gets a
fresh model and tinygrad's BEAM search runs for real (`search_s`), then prefill and decode are timed.
The best width is the one with the lowest ms/token. Jobs run one at a time, and pause or resume takes
effect after the current width finishes.

**Auto-optimize** measures width 0, then 1, 2, 3, … and keeps going while `auto` is on
(`POST /api/benchmarks/{job}/auto` with `{"enabled": true|false}`). Turning it off lets the current width
finish, and the job then ends as `done`. It always stops at width 32 (`AUTO_MAX_WIDTH`). Kernel options come from tinygrad's own cache
(`~/Library/Caches/tinygrad/cache.db`, table `beam_search_*`) and are copied into the app database.

## Threading (why it looks unusual)

- The **main thread** runs all tinygrad work (`gpu_worker.py`). BEAM search installs signal handlers, which
  only works on the main thread, and Metal objects crash when a short-lived thread exits.
- The internal API and the OpenAI-compatible API run on background threads and hand their model work to
  the main thread.
- `PARALLEL=0` is set, so tinygrad compiles kernels in-process. Its worker pool forks, and forking after
  Metal starts crashes. Compilation is slower as a result.
- The process exits with `os._exit` after shutdown, to avoid a crash in Metal teardown.
- `python/scripts/llm_model.py` is a vendored copy of tinygrad's dense Llama/Qwen2 transformer
  (MIT). The published tinygrad 0.14.0 wheel is missing `tinygrad/llm/kernels`, which its `llm/model.py`
  imports.

## Setup

1. Unpack a python-build-standalone runtime directly into `python/`, so that `python/bin/python3` exists on macOS/Linux and `python/python.exe` on Windows. The setup used `cpython-3.12.16+20261110-aarch64-apple-darwin-install_only.tar.gz` from [astral-sh/python-build-standalone](https://github.com/astral-sh/python-build-standalone/releases/tag/20261110) (macOS Apple Silicon). Other platforms need their matching archive.
2. Create the venv and install dependencies:
   ```sh
   python/bin/python3 -m venv python/venv
   python/venv/bin/pip install -r python/requirements.txt
   ```
   Windows: `python\python.exe -m venv python\venv` then `python\venv\Scripts\pip install -r python\requirements.txt`.
3. `npm install`
4. `npm start` to run in development.
5. `npm run dist` to build installers.

## Things to check before shipping

- **Relocatable venv.** A venv records its base interpreter path in `pyvenv.cfg`. If `python/venv` is created in the project and then copied into the app, it can break. The safer option is to skip the venv and install the requirements into the standalone interpreter itself: `python/bin/python3 -m pip install -r python/requirements.txt`. Then point `resolvePython()` in `electron/main.js` at `python/bin/python3` only.
- **Native code signing.** Bundled Python binaries and `.so`/`.dylib` files need signing for macOS notarization and may need `electron-builder` `mac.binaries` / entitlements config.
- **Electron version.** `package.json` pins `electron` `^33` and `electron-builder` `^25` as a starting point. Update to current releases.
- **tinygrad device.** `Device.DEFAULT` decides which backend runs (CPU, METAL, CUDA, …). Check it on each target platform.
- **Dev vs. packaged paths.** Dev uses `./python`; packaged uses `process.resourcesPath/python`.
