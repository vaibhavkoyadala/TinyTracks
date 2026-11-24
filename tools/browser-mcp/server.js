#!/usr/bin/env node
// MCP server (stdio) that exposes the TinyTracks browser pane as tools.
// It forwards each call to the running app's browser-control endpoint, which reads the
// port and token from browser-control.json in the app's data directory. The app must be running.
//
// No dependencies: speaks newline-delimited JSON-RPC 2.0 as MCP requires.

const fs = require('node:fs');
const path = require('node:path');

const DATA_DIR =
  process.env.TINYTRACKS_DATA_DIR || path.join(__dirname, '..', '..', 'python', 'data');
const CONTROL_FILE = path.join(DATA_DIR, 'browser-control.json');
const PROTOCOL_VERSION = '2024-11-05';

const tabProp = {
  type: 'integer',
  description: 'Tab id from browser_list_tabs.',
};
const selectorProp = {
  type: 'string',
  description: 'CSS selector, for example "#search" or "button.primary".',
};

const TOOLS = [
  {
    command: 'list_tabs',
    description: 'List the open browser tabs with their ids, URLs, titles and loading state.',
    properties: {},
  },
  {
    command: 'navigate',
    description: 'Load a URL in a tab. Only http, https and about:blank are allowed.',
    properties: { tab: tabProp, url: { type: 'string', description: 'Address to load.' } },
    required: ['tab', 'url'],
  },
  {
    command: 'back',
    description: 'Go back one page in a tab.',
    properties: { tab: tabProp },
    required: ['tab'],
  },
  {
    command: 'forward',
    description: 'Go forward one page in a tab.',
    properties: { tab: tabProp },
    required: ['tab'],
  },
  {
    command: 'reload',
    description: 'Reload a tab.',
    properties: { tab: tabProp },
    required: ['tab'],
  },
  {
    command: 'text',
    description: "Return the visible text of a tab's page (document.body.innerText).",
    properties: { tab: tabProp },
    required: ['tab'],
  },
  {
    command: 'html',
    description: "Return the page's current HTML (document.documentElement.outerHTML).",
    properties: { tab: tabProp },
    required: ['tab'],
  },
  {
    command: 'eval',
    description:
      'Run JavaScript in the page and return its JSON-serialisable result. Runs inside the page, not in the app.',
    properties: {
      tab: tabProp,
      expression: { type: 'string', description: 'JavaScript expression or statement block; promises are awaited.' },
    },
    required: ['tab', 'expression'],
  },
  {
    command: 'click',
    description: 'Scroll an element into view and click it with a real mouse event at its centre.',
    properties: { tab: tabProp, selector: selectorProp },
    required: ['tab', 'selector'],
  },
  {
    command: 'type',
    description: 'Focus an input and type text into it (inserted at the caret; does not clear existing text).',
    properties: {
      tab: tabProp,
      selector: selectorProp,
      text: { type: 'string', description: 'Text to type.' },
    },
    required: ['tab', 'selector', 'text'],
  },
  {
    command: 'press_key',
    description:
      'Press a key in the focused element. Accepts a single character or one of: Enter, Tab, Escape, Backspace, Delete, ArrowLeft/Right/Up/Down, Home, End, PageUp, PageDown, Space.',
    properties: { tab: tabProp, key: { type: 'string', description: 'Key name.' } },
    required: ['tab', 'key'],
  },
  {
    command: 'wait_for',
    description: 'Wait until an element matching the selector exists in the page.',
    properties: {
      tab: tabProp,
      selector: selectorProp,
      timeout_ms: { type: 'integer', description: 'Give up after this many ms (default 10000, max 60000).' },
    },
    required: ['tab', 'selector'],
  },
  {
    command: 'screenshot',
    description: 'Capture the tab as a PNG. The tab should be the visible one in the browser pane.',
    properties: { tab: tabProp },
    required: ['tab'],
  },
  {
    command: 'cdp',
    description:
      'Raw Chrome DevTools Protocol call on a tab, for anything the other tools do not cover. Browser and Target domains are refused.',
    properties: {
      tab: tabProp,
      method: { type: 'string', description: 'CDP method, for example "Page.getLayoutMetrics".' },
      params: { type: 'object', description: 'Method parameters.' },
    },
    required: ['tab', 'method'],
  },
];

const TOOL_BY_NAME = new Map(TOOLS.map((t) => [`browser_${t.command}`, t]));

function toolList() {
  return TOOLS.map((t) => ({
    name: `browser_${t.command}`,
    description: t.description,
    inputSchema: {
      type: 'object',
      properties: t.properties,
      required: t.required || [],
      additionalProperties: false,
    },
  }));
}

class ToolError extends Error {}

async function callApp(command, args) {
  let info;
  try {
    info = JSON.parse(fs.readFileSync(CONTROL_FILE, 'utf8'));
  } catch {
    throw new ToolError(
      `TinyTracks is not running (no ${CONTROL_FILE}). Start the app and try again.`
    );
  }
  let res;
  try {
    res = await fetch(`http://127.0.0.1:${info.port}/browser/${command}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${info.token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(args || {}),
    });
  } catch {
    throw new ToolError('TinyTracks is not reachable. The app may have stopped; start it and try again.');
  }
  const body = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
  if (!body.ok) throw new ToolError(body.error || `HTTP ${res.status}`);
  return body.result;
}

async function callTool(name, args) {
  const tool = TOOL_BY_NAME.get(name);
  if (!tool) throw new ToolError(`Unknown tool ${name}`);
  const result = await callApp(tool.command, args);
  if (tool.command === 'screenshot' && result && result.png_base64) {
    return { content: [{ type: 'image', data: result.png_base64, mimeType: 'image/png' }] };
  }
  return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
}

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

async function handle(msg) {
  const { id, method, params } = msg;
  const isRequest = id !== undefined;

  if (method === 'notifications/initialized' || method === 'notifications/cancelled') return;

  if (!isRequest) return;

  try {
    let result;
    switch (method) {
      case 'initialize':
        result = {
          protocolVersion: PROTOCOL_VERSION,
          capabilities: { tools: {} },
          serverInfo: { name: 'tinytracks-browser', version: '0.1.0' },
        };
        break;
      case 'ping':
        result = {};
        break;
      case 'tools/list':
        result = { tools: toolList() };
        break;
      case 'tools/call': {
        try {
          result = await callTool(params.name, params.arguments || {});
        } catch (err) {
          if (!(err instanceof ToolError)) throw err;
          result = { content: [{ type: 'text', text: err.message }], isError: true };
        }
        break;
      }
      default:
        send({ jsonrpc: '2.0', id, error: { code: -32601, message: `Method not found: ${method}` } });
        return;
    }
    send({ jsonrpc: '2.0', id, result });
  } catch (err) {
    send({ jsonrpc: '2.0', id, error: { code: -32603, message: err.message } });
  }
}

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  let nl;
  while ((nl = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, nl).trim();
    buffer = buffer.slice(nl + 1);
    if (!line) continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      send({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } });
      continue;
    }
    handle(msg);
  }
});
