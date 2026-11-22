const { contextBridge, ipcRenderer } = require('electron');

// The only surface the web page gets. It can call the backend API and read status; nothing else.
contextBridge.exposeInMainWorld('backend', {
  status: () => ipcRenderer.invoke('backend:status'),
  request: (method, path, body) => ipcRenderer.invoke('backend:request', { method, path, body }),
});
