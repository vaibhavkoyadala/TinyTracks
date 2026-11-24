# TinyTracks browser tools (MCP)

Lets an editor or agent drive the browser pane in TinyTracks: list tabs, navigate, read text,
run JavaScript, click, type, press keys, wait for elements, take screenshots, and send raw
Chrome DevTools Protocol commands.

The TinyTracks app must be running. The tools talk to it through a local endpoint on
127.0.0.1 that requires a token; the app writes the port and token to `browser-control.json`
in its data directory each time it starts.

## Connect an editor

Most MCP clients (Claude Code, Cursor, VS Code) read a JSON config like this. Use the absolute
path to `server.js` in this repo:

```json
{
  "mcpServers": {
    "tinytracks-browser": {
      "command": "node",
      "args": ["/absolute/path/to/TinyTracks/tools/browser-mcp/server.js"]
    }
  }
}
```

If you run a packaged build, the data directory is elsewhere. Point the server at it with
`TINYTRACKS_DATA_DIR`, which should be the same folder the app uses (the userData `data` folder).

## Tools

| Tool | Arguments | What it does |
| --- | --- | --- |
| `browser_list_tabs` | none | Tab ids, URLs, titles, loading state |
| `browser_navigate` | `tab`, `url` | Load http, https or about:blank |
| `browser_back` / `browser_forward` / `browser_reload` | `tab` | History and reload |
| `browser_text` | `tab` | Visible text of the page |
| `browser_html` | `tab` | Current HTML |
| `browser_eval` | `tab`, `expression` | Run JavaScript in the page; returns the JSON result |
| `browser_click` | `tab`, `selector` | Real mouse click at the element's centre |
| `browser_type` | `tab`, `selector`, `text` | Focus and type into an input |
| `browser_press_key` | `tab`, `key` | Enter, Tab, Escape, arrows, or one character |
| `browser_wait_for` | `tab`, `selector`, `timeout_ms` | Wait for an element to appear |
| `browser_screenshot` | `tab` | PNG of the tab |
| `browser_cdp` | `tab`, `method`, `params` | Raw CDP call (Browser and Target domains refused) |

## Notes

- Tab ids come from `browser_list_tabs`. They change when the app restarts.
- Screenshots and clicks work best on the tab that is visible in the pane.
- Pages run in a sandbox with no access to the app's backend. The tools only reach the page
  itself; `browser_eval` runs inside the page, not in the app.
- Only http, https and about:blank can be loaded, the same rule as the address bar.
