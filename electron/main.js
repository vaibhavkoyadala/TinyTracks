const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const net = require('node:net');
const path = require('node:path');

const isPackaged = app.isPackaged;
// In dev, python/ sits next to electron/. When packaged, extraResources copies it to resources/python.
const pythonRoot = isPackaged
  ? path.join(process.resourcesPath, 'python')
  : path.join(__dirname, '..', 'python');

// Shared secret between Electron and the Python server. Only the main process knows it;
// the renderer reaches the API through IPC, so it never sees the token.
const apiToken = crypto.randomBytes(32).toString('hex');

// Where the database and model weights live. A packaged app's resources folder is read-only
// and replaced on update, so user data goes to the per-user app data folder instead.
function dataDir() {
  return isPackaged ? path.join(app.getPath('userData'), 'data') : path.join(pythonRoot, 'data');
}

let pyProc = null;
let apiBase = null;
let backendReady = false;
let backendError = null;
let quitting = false;

function resolvePython() {
  const candidates =
    process.platform === 'win32'
      ? [path.join(pythonRoot, 'venv', 'Scripts', 'python.exe'), path.join(pythonRoot, 'python.exe')]
      : [path.join(pythonRoot, 'venv', 'bin', 'python3'), path.join(pythonRoot, 'bin', 'python3')];
  const found = candidates.find((p) => fs.existsSync(p));
  if (!found) {
    throw new Error(`Python runtime not found. Looked in:\n${candidates.join('\n')}`);
  }
  return found;
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

async function startBackend() {
  const port = await freePort();
  apiBase = `http://127.0.0.1:${port}`;
  const script = path.join(pythonRoot, 'scripts', 'worker.py');

  pyProc = spawn(
    resolvePython(),
    [script, '--host', '127.0.0.1', '--port', String(port)],
    {
      env: {
        ...process.env,
        TINYTRACKS_TOKEN: apiToken,
        TINYTRACKS_DATA_DIR: dataDir(),
        PYTHONUNBUFFERED: '1',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    }
  );

  pyProc.stdout.on('data', (d) => process.stdout.write(`[python] ${d}`));
  pyProc.stderr.on('data', (d) => process.stderr.write(`[python] ${d}`));

  pyProc.on('exit', (code, signal) => {
    backendReady = false;
    pyProc = null;
    if (quitting) return;
    backendError = `Python server exited (code ${code}, signal ${signal})`;
    dialog.showErrorBox('Backend stopped', backendError);
  });

  await waitForHealth(60_000);
  backendReady = true;
}

async function waitForHealth(timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!pyProc) throw new Error('Python server failed to start');
    try {
      const res = await fetch(`${apiBase}/api/health`, {
        headers: { 'X-TinyTracks-Token': apiToken },
      });
      if (res.ok) return;
    } catch {
      // not listening yet
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error('Timed out waiting for the Python server');
}

function registerIpc() {
  ipcMain.handle('backend:status', () => ({
    ready: backendReady,
    error: backendError,
  }));

  ipcMain.handle('backend:request', async (_event, { method, path: apiPath, body }) => {
    if (!backendReady) throw new Error(backendError || 'Backend is not ready yet');
    if (typeof apiPath !== 'string' || !apiPath.startsWith('/api/')) {
      throw new Error('Only /api/* paths are allowed');
    }
    const res = await fetch(`${apiBase}${apiPath}`, {
      method: method || 'GET',
      headers: {
        'X-TinyTracks-Token': apiToken,
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      const detail = data && data.detail ? data.detail : res.statusText;
      throw new Error(`${res.status}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
    }
    return data;
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
}

app.whenReady().then(async () => {
  registerIpc();
  createWindow();
  try {
    await startBackend();
  } catch (err) {
    backendError = err.message;
    dialog.showErrorBox('Backend failed to start', backendError);
  }
});

app.on('window-all-closed', () => {
  app.quit();
});

app.on('before-quit', () => {
  quitting = true;
  if (pyProc) pyProc.kill();
});
