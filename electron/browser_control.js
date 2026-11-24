// Browser control: drives the browser pane's tabs over the Chrome DevTools Protocol (CDP).
// Runs in the Electron main process. Exposed on a local, token-gated HTTP endpoint; the MCP
// server in tools/browser-mcp/ turns each command into a tool that an editor can call.

const crypto = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { webContents } = require('electron');

const MAX_BODY = 1_000_000;
const MAX_TEXT = 200_000;
const DENIED_CDP_DOMAINS = new Set(['Browser', 'Target']); // would reach outside the page

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function browserTabs() {
  return webContents
    .getAllWebContents()
    .filter((c) => c.getType() === 'webview' && !c.isDestroyed());
}

function findTab(id) {
  const tab = browserTabs().find((c) => c.id === Number(id));
  if (!tab) throw new HttpError(404, `No browser tab with id ${id}. Call browser_list_tabs first.`);
  return tab;
}

// Same rule as the address bar: http(s) and about:blank only.
function checkUrl(url) {
  if (typeof url !== 'string' || !url.trim()) throw new HttpError(400, 'url is required');
  const u = url.trim();
  if (u === 'about:blank' || /^https?:\/\//i.test(u)) return u;
  throw new HttpError(400, 'Only http, https and about:blank URLs are allowed');
}

function requireString(value, name) {
  if (typeof value !== 'string' || !value) throw new HttpError(400, `${name} is required`);
  return value;
}

async function cdp(tab, method, params = {}) {
  if (!tab.debugger.isAttached()) {
    try {
      tab.debugger.attach('1.3');
    } catch (err) {
      throw new HttpError(409, `Cannot attach to the tab: ${err.message}`);
    }
  }
  return tab.debugger.sendCommand(method, params);
}

async function evaluate(tab, expression) {
  const res = await cdp(tab, 'Runtime.evaluate', {
    expression,
    returnByValue: true,
    awaitPromise: true,
    userGesture: true,
  });
  if (res.exceptionDetails) {
    const ex = res.exceptionDetails.exception;
    throw new HttpError(422, (ex && ex.description) || res.exceptionDetails.text);
  }
  return res.result.value;
}

function truncate(text) {
  if (typeof text !== 'string') return '';
  return text.length > MAX_TEXT ? `${text.slice(0, MAX_TEXT)}\n[truncated at ${MAX_TEXT} characters]` : text;
}

// Finds an element, scrolls it into view, and returns its centre in CSS pixels.
async function centreOf(tab, selector) {
  const point = await evaluate(
    tab,
    `(() => {
      const el = document.querySelector(${JSON.stringify(selector)});
      if (!el) return null;
      el.scrollIntoView({ block: 'center', inline: 'center' });
      const r = el.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    })()`
  );
  if (!point) throw new HttpError(404, `No element matches ${selector}`);
  return point;
}

const KEYS = {
  Enter: { code: 'Enter', keyCode: 13, text: '\r' },
  Tab: { code: 'Tab', keyCode: 9 },
  Escape: { code: 'Escape', keyCode: 27 },
  Backspace: { code: 'Backspace', keyCode: 8 },
  Delete: { code: 'Delete', keyCode: 46 },
  ArrowLeft: { code: 'ArrowLeft', keyCode: 37 },
  ArrowUp: { code: 'ArrowUp', keyCode: 38 },
  ArrowRight: { code: 'ArrowRight', keyCode: 39 },
  ArrowDown: { code: 'ArrowDown', keyCode: 40 },
  Home: { code: 'Home', keyCode: 36 },
  End: { code: 'End', keyCode: 35 },
  PageUp: { code: 'PageUp', keyCode: 33 },
  PageDown: { code: 'PageDown', keyCode: 34 },
  Space: { code: 'Space', keyCode: 32, text: ' ' },
};

function keyDefinition(key) {
  if (KEYS[key]) return { key, ...KEYS[key] };
  if (key.length === 1) {
    const upper = key.toUpperCase();
    const isLetter = upper >= 'A' && upper <= 'Z';
    const isDigit = key >= '0' && key <= '9';
    return {
      key,
      text: key,
      code: isLetter ? `Key${upper}` : isDigit ? `Digit${key}` : '',
      keyCode: isLetter ? upper.charCodeAt(0) : isDigit ? key.charCodeAt(0) : 0,
    };
  }
  throw new HttpError(400, `Unknown key "${key}". Use a single character or one of: ${Object.keys(KEYS).join(', ')}`);
}

// Every command takes its arguments as one JSON object and returns a JSON-serialisable result.
// Screenshot returns { png_base64 }; the MCP layer turns it into an image.
const commands = {
  async list_tabs() {
    return browserTabs().map((t) => ({
      tab: t.id,
      url: t.getURL(),
      title: t.getTitle(),
      loading: t.isLoading(),
    }));
  },

  async navigate({ tab, url }) {
    const wc = findTab(tab);
    await wc.loadURL(checkUrl(url));
    return { tab: wc.id, url: wc.getURL(), title: wc.getTitle() };
  },

  async back({ tab }) {
    const wc = findTab(tab);
    if (!wc.canGoBack()) throw new HttpError(409, 'Nothing to go back to');
    wc.goBack();
    return { url: wc.getURL() };
  },

  async forward({ tab }) {
    const wc = findTab(tab);
    if (!wc.canGoForward()) throw new HttpError(409, 'Nothing to go forward to');
    wc.goForward();
    return { url: wc.getURL() };
  },

  async reload({ tab }) {
    const wc = findTab(tab);
    wc.reload();
    return { url: wc.getURL() };
  },

  async text({ tab }) {
    const wc = findTab(tab);
    return { text: truncate(await evaluate(wc, 'document.body ? document.body.innerText : ""')) };
  },

  async html({ tab }) {
    const wc = findTab(tab);
    return { html: truncate(await evaluate(wc, 'document.documentElement.outerHTML')) };
  },

  async eval({ tab, expression }) {
    const wc = findTab(tab);
    const value = await evaluate(wc, requireString(expression, 'expression'));
    return { value: value === undefined ? null : value };
  },

  async click({ tab, selector }) {
    const wc = findTab(tab);
    const { x, y } = await centreOf(wc, requireString(selector, 'selector'));
    const base = { x, y, button: 'left', clickCount: 1 };
    await cdp(wc, 'Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
    await cdp(wc, 'Input.dispatchMouseEvent', { type: 'mousePressed', ...base });
    await cdp(wc, 'Input.dispatchMouseEvent', { type: 'mouseReleased', ...base });
    return { clicked: selector };
  },

  // Focuses the element and inserts text at the caret, so it appends to whatever is there.
  async type({ tab, selector, text }) {
    const wc = findTab(tab);
    requireString(selector, 'selector');
    if (typeof text !== 'string') throw new HttpError(400, 'text is required');
    const focused = await evaluate(
      wc,
      `(() => {
        const el = document.querySelector(${JSON.stringify(selector)});
        if (!el) return false;
        el.scrollIntoView({ block: 'center', inline: 'center' });
        el.focus();
        return true;
      })()`
    );
    if (!focused) throw new HttpError(404, `No element matches ${selector}`);
    await cdp(wc, 'Input.insertText', { text });
    return { typed: text.length };
  },

  async press_key({ tab, key }) {
    const wc = findTab(tab);
    const def = keyDefinition(requireString(key, 'key'));
    const base = {
      key: def.key,
      code: def.code,
      windowsVirtualKeyCode: def.keyCode || 0,
      nativeVirtualKeyCode: def.keyCode || 0,
    };
    await cdp(wc, 'Input.dispatchKeyEvent', {
      type: def.text ? 'keyDown' : 'rawKeyDown',
      ...base,
      text: def.text,
    });
    await cdp(wc, 'Input.dispatchKeyEvent', { type: 'keyUp', ...base });
    return { pressed: def.key };
  },

  async wait_for({ tab, selector, timeout_ms }) {
    const wc = findTab(tab);
    requireString(selector, 'selector');
    const limit = Math.min(Math.max(Number(timeout_ms) || 10_000, 100), 60_000);
    const deadline = Date.now() + limit;
    const expression = `!!document.querySelector(${JSON.stringify(selector)})`;
    while (Date.now() < deadline) {
      if (await evaluate(wc, expression)) return { found: selector };
      await new Promise((r) => setTimeout(r, 250));
    }
    throw new HttpError(408, `${selector} did not appear within ${limit} ms`);
  },

  async screenshot({ tab }) {
    const wc = findTab(tab);
    const { data } = await cdp(wc, 'Page.captureScreenshot', { format: 'png' });
    return { png_base64: data };
  },

  // Raw CDP passthrough for anything the commands above do not cover.
  async cdp({ tab, method, params }) {
    const wc = findTab(tab);
    const name = requireString(method, 'method');
    const domain = name.split('.')[0];
    if (DENIED_CDP_DOMAINS.has(domain)) throw new HttpError(403, `The ${domain} domain is not available`);
    return cdp(wc, name, params && typeof params === 'object' ? params : {});
  },
};

function constantTimeEqual(a, b) {
  const x = Buffer.from(a);
  const y = Buffer.from(b);
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY) {
        reject(new HttpError(413, 'Request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      const text = Buffer.concat(chunks).toString('utf8');
      if (!text.trim()) return resolve({});
      try {
        resolve(JSON.parse(text));
      } catch {
        reject(new HttpError(400, 'Body must be JSON'));
      }
    });
    req.on('error', reject);
  });
}

// Starts the endpoint on 127.0.0.1 with a fresh token. The port and token go to
// <dir>/browser-control.json (owner-only), so the MCP server can find the running app.
function startBrowserControl(dir) {
  const token = crypto.randomBytes(32).toString('hex');

  const server = http.createServer(async (req, res) => {
    const send = (status, body) => {
      res.writeHead(status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(body));
    };
    try {
      const auth = req.headers.authorization || '';
      if (!constantTimeEqual(auth, `Bearer ${token}`)) return send(401, { ok: false, error: 'unauthorized' });
      const match = /^\/browser\/([a-z_]+)$/.exec(req.url.split('?')[0]);
      if (req.method !== 'POST' || !match) return send(404, { ok: false, error: 'not found' });
      const command = commands[match[1]];
      if (!command) return send(404, { ok: false, error: `unknown command ${match[1]}` });
      const args = await readJson(req);
      const result = await command(args);
      send(200, { ok: true, result: result === undefined ? null : result });
    } catch (err) {
      if (!(err instanceof HttpError)) console.error('[browser-control]', err);
      send(err.status || 500, { ok: false, error: err.message });
    }
  });

  server.listen(0, '127.0.0.1', () => {
    const { port } = server.address();
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, 'browser-control.json');
    fs.writeFileSync(file, JSON.stringify({ port, token, pid: process.pid }, null, 2), { mode: 0o600 });
    console.log(`[browser-control] listening on 127.0.0.1:${port}`);
  });

  return {
    stop() {
      server.close();
      fs.rmSync(path.join(dir, 'browser-control.json'), { force: true });
    },
  };
}

module.exports = { startBrowserControl, commands };
