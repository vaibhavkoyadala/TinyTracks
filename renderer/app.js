const $ = (id) => document.getElementById(id);

// Small DOM helper. Server text always goes in via textContent, never innerHTML.
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'on') for (const [ev, fn] of Object.entries(v)) node.addEventListener(ev, fn);
    else if (k === 'disabled') node.disabled = Boolean(v);
    else if (k === 'value') node.value = v;
    else node.setAttribute(k, v);
  }
  node.append(...children);
  return node;
}

const api = (method, path, body) => window.backend.request(method, path, body);

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  const units = ['KB', 'MB', 'GB'];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 2 : 1)} ${units[i]}`;
}

function formatWhen(epochSeconds) {
  return new Date(epochSeconds * 1000).toLocaleTimeString();
}

// ----- tabs ---------------------------------------------------------------------

let activeTab = 'dashboard';

document.querySelectorAll('.tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    activeTab = btn.dataset.tab;
    document.querySelectorAll('.tab').forEach((b) => {
      b.classList.toggle('active', b === btn);
      b.setAttribute('aria-selected', String(b === btn));
    });
    document.querySelectorAll('.tab-body').forEach((s) => {
      s.classList.toggle('active', s.id === `tab-${activeTab}`);
    });
    refreshActiveTab();
  });
});

// ----- backend status ---------------------------------------------------------------

async function refreshStatus() {
  const s = await window.backend.status();
  const statusEl = $('status');
  if (s.ready) {
    statusEl.textContent = 'Ready';
    statusEl.className = 'status ok';
  } else if (s.error) {
    statusEl.textContent = `Error: ${s.error}`;
    statusEl.className = 'status err';
  } else {
    statusEl.textContent = 'Starting backend…';
    statusEl.className = 'status';
  }
  return s.ready;
}

// ----- compute & health -------------------------------------------------------------

async function loadHealth() {
  try {
    const data = await api('GET', '/api/health');
    $('health-out').textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    $('health-out').textContent = err.message;
  }
}

$('compute-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const out = $('compute-out');
  const values = $('values').value
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean)
    .map(Number);
  out.textContent = 'Running…';
  try {
    const data = await api('POST', '/api/compute', { values });
    out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    out.textContent = err.message;
  }
});

// ----- models -------------------------------------------------------------------------

const DOWNLOAD_LABEL = {
  not_downloaded: 'NOT DOWNLOADED',
  downloading: 'DOWNLOADING',
  paused: 'PAUSED',
  verifying: 'CHECKING SHA-256',
  ready: 'READY',
  failed: 'FAILED',
};

const SERVING_LABEL = { downloaded: 'OFF', loaded: 'LOADED · PAUSED', live: 'LIVE' };

const rows = new Map();     // model id -> row refs
const samples = new Map();  // model id -> { bytes, time } for speed
let modelList = [];         // latest /api/models payload

function setMsg(text, isError = false) {
  const m = $('models-msg');
  m.textContent = text;
  m.className = isError ? 'msg error' : 'msg';
}

function buildModelRow(model) {
  const bar = el('div', { class: 'bar' }, el('div', { class: 'fill' }));
  const dlBtn = el('button', { type: 'button', on: { click: () => row.onDownload && row.onDownload() } });
  const delBtn = el('button', {
    type: 'button', text: 'DELETE', on: { click: () => onDeleteFiles(model.id, model.name) },
  });
  const transport = [
    el('button', { type: 'button', text: 'LOAD', on: { click: () => call('POST', `/api/models/${model.id}/load`, 'Loading…') } }),
    el('button', { type: 'button', text: 'PLAY', on: { click: () => call('POST', `/api/models/${model.id}/play`, 'Starting…') } }),
    el('button', { type: 'button', text: 'PAUSE', on: { click: () => call('POST', `/api/models/${model.id}/pause`, 'Pausing…') } }),
    el('button', { type: 'button', text: 'EJECT', on: { click: () => onEject(model.id, model.name) } }),
  ];
  const row = {
    tr: el('tr', { 'data-id': model.id },
      el('td', {},
        el('div', { class: 'model-name', text: model.name }),
        el('div', { class: 'sub', text: `${model.description} (${model.file_count} file${model.file_count === 1 ? '' : 's'})` }),
        el('div', { class: 'sub mono', text: model.architecture }),
      ),
      el('td', { class: 'mono', text: formatBytes(model.size_bytes) }),
      el('td', {},
        el('div', { class: 'mono st' }),
        bar,
        el('div', { class: 'mono sub progress-line' }),
        el('div', { class: 'err-text' }),
        el('div', { class: 'btn-row' }, dlBtn, delBtn),
      ),
      el('td', {},
        el('div', { class: 'mono st serving-state' }),
        el('div', { class: 'sub serving-reason' }),
      ),
      el('td', {}, el('div', { class: 'btn-row' }, ...transport)),
    ),
    fill: bar.firstChild,
    statusEl: null,
    errEl: null,
    progressEl: null,
    servingEl: null,
    reasonEl: null,
    dlBtn,
    delBtn,
    transport,
    onDownload: null,
  };
  row.statusEl = row.tr.querySelector('.st');
  row.errEl = row.tr.querySelector('.err-text');
  row.progressEl = row.tr.querySelector('.progress-line');
  row.servingEl = row.tr.querySelector('.serving-state');
  row.reasonEl = row.tr.querySelector('.serving-reason');
  return row;
}

function updateModelRow(row, m) {
  // Download column
  row.statusEl.textContent = DOWNLOAD_LABEL[m.status] || m.status.toUpperCase();
  row.statusEl.className = `mono st st-${m.status}`;
  row.errEl.textContent = m.error && m.status !== 'ready' ? m.error : '';
  row.errEl.title = m.error || '';
  row.fill.style.width = `${m.percent}%`;

  let speed = '';
  const now = Date.now();
  const prev = samples.get(m.id);
  if (m.status === 'downloading' && prev && now > prev.time && m.bytes_done >= prev.bytes) {
    speed = ` · ${formatBytes(((m.bytes_done - prev.bytes) * 1000) / (now - prev.time))}/s`;
  }
  samples.set(m.id, { bytes: m.bytes_done, time: now });
  row.progressEl.textContent = `${m.percent.toFixed(1)}%  ${formatBytes(m.bytes_done)} / ${formatBytes(m.size_bytes)}${speed}`;

  const busy = m.status === 'verifying';
  row.dlBtn.disabled = busy || m.status === 'ready';
  if (m.status === 'downloading') {
    row.dlBtn.textContent = 'PAUSE';
    row.onDownload = () => call('POST', `/api/models/${m.id}/download/pause`, 'Pausing download…');
  } else if (m.status === 'verifying') {
    row.dlBtn.textContent = 'CHECKING…';
    row.onDownload = null;
  } else if (m.status === 'ready') {
    row.dlBtn.textContent = 'DOWNLOADED';
    row.onDownload = null;
  } else {
    row.dlBtn.textContent = m.status === 'failed' ? 'RETRY' : m.bytes_done > 0 ? 'RESUME' : 'DOWNLOAD';
    row.onDownload = () => call('POST', `/api/models/${m.id}/download`, 'Starting download…');
  }
  // Files can only be deleted while the model is not loaded (eject first) and no transfer is running.
  const transferring = m.status === 'downloading' || m.status === 'verifying';
  row.delBtn.disabled = transferring || m.state !== 'downloaded' || (m.status !== 'ready' && m.bytes_done === 0);

  // Serving column and transport buttons
  row.servingEl.textContent = SERVING_LABEL[m.state] || m.state.toUpperCase();
  row.servingEl.className = `mono st serving-${m.state}`;
  row.reasonEl.textContent = m.servable ? '' : `Not servable yet (${m.architecture})`;
  const [loadBtn, playBtn, pauseBtn, ejectBtn] = row.transport;
  const canServe = m.status === 'ready' && m.servable;
  loadBtn.disabled = !canServe || m.state !== 'downloaded';
  playBtn.disabled = !canServe || m.state === 'live';
  pauseBtn.disabled = m.state !== 'live';
  ejectBtn.disabled = m.state === 'downloaded';
  for (const b of row.transport) {
    b.title = canServe ? '' : 'Download the model first. Only Llama and Qwen2 models can be served.';
  }
}

async function onDeleteFiles(id, name) {
  if (!window.confirm(`Delete the downloaded files for "${name}"?`)) return;
  await call('DELETE', `/api/models/${id}`, 'Deleting…');
}

async function onEject(id, name) {
  if (!window.confirm(`Eject "${name}" from memory? It stops serving until loaded again.`)) return;
  await call('POST', `/api/models/${id}/eject`, 'Ejecting…');
}

// Run an action, show the outcome, then refresh.
async function call(method, path, label) {
  setMsg(label);
  try {
    await api(method, path);
    setMsg('');
  } catch (err) {
    setMsg(err.message, true);
  }
  await refreshModels();
}

async function refreshModels() {
  let data;
  try {
    data = await api('GET', '/api/models');
  } catch (err) {
    setMsg(err.message, true);
    return;
  }
  modelList = data.models;

  const body = $('models-body');
  const seen = new Set();
  for (const m of data.models) {
    seen.add(m.id);
    let row = rows.get(m.id);
    if (!row) {
      row = buildModelRow(m);
      rows.set(m.id, row);
      body.append(row.tr);
    }
    updateModelRow(row, m);
  }
  for (const [id, row] of rows) {
    if (!seen.has(id)) {
      row.tr.remove();
      rows.delete(id);
    }
  }

  const installed = data.models.filter((m) => m.status === 'ready').length;
  const live = data.models.filter((m) => m.state === 'live').length;
  const active = data.models.filter((m) => m.status === 'downloading' || m.status === 'verifying').length;
  $('models-foot').textContent =
    `${data.models.length} MODELS  ·  ${installed} DOWNLOADED  ·  ${live} LIVE  ·  ${active} DOWNLOADING`;

  populateModelSelects();
}

let optionSignature = '';

// Rebuild the dropdowns only when the set of choices changes, so an open dropdown is not reset every second.
function populateModelSelects() {
  const servable = modelList.filter((m) => m.status === 'ready' && m.servable);
  const signature = `${servable.map((m) => m.id).join(',')}|${modelList.map((m) => m.id).join(',')}`;
  if (signature === optionSignature) return;
  optionSignature = signature;

  const beamSelect = $('beam-model');
  const previousBeam = beamSelect.value;
  beamSelect.replaceChildren(...servable.map((m) => el('option', { value: m.id, text: m.name })));
  if (servable.some((m) => m.id === previousBeam)) beamSelect.value = previousBeam;

  const kernelSelect = $('kernel-model');
  const previousKernel = kernelSelect.value;
  kernelSelect.replaceChildren(...modelList.map((m) => el('option', { value: m.id, text: m.name })));
  if (modelList.some((m) => m.id === previousKernel)) kernelSelect.value = previousKernel;
}

// ----- dashboard ----------------------------------------------------------------------

const liveCards = new Map(); // model id -> refs

function buildLiveCard(model) {
  const copy = el('button', { type: 'button', text: 'COPY' });
  const pre = el('pre', { class: 'curl' });
  const stats = el('div', { class: 'mono sub stats' });
  const card = el('div', { class: 'live-card' },
    el('div', { class: 'live-head' },
      el('span', { class: 'model-name', text: model.name }),
      el('span', { class: 'mono sub', text: model.id }),
    ),
    el('div', { class: 'kv' }, el('span', { text: 'Endpoint' }), el('code', { class: 'url', text: model.url })),
    el('div', { class: 'curl-head' }, el('span', { class: 'sub', text: 'Sample request' }), copy),
    pre,
    stats,
  );
  const refs = { card, pre, stats, copy, curl: '' };
  copy.addEventListener('click', () => copyText(refs.curl, copy));
  return refs;
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = 'COPIED';
  } catch {
    button.textContent = 'FAILED';
  }
  setTimeout(() => { button.textContent = 'COPY'; }, 1200);
}

async function refreshDashboard() {
  let data;
  try {
    data = await api('GET', '/api/serving');
  } catch (err) {
    $('api-listener').textContent = err.message;
    return;
  }
  $('api-base').textContent = data.base_url;
  $('api-key').textContent = data.api_key;
  $('api-listener').textContent = data.openai.running
    ? `Listening on ${data.base_url.replace('/v1', '')}`
    : `Not running: ${data.openai.error}`;
  $('api-listener').className = data.openai.running ? 'st-ready' : 'st-failed';

  const live = data.models.filter((m) => m.live);
  $('live-empty').hidden = live.length > 0;
  const list = $('live-list');
  const liveIds = new Set(live.map((m) => m.id));

  for (const model of live) {
    let refs = liveCards.get(model.id);
    if (!refs) {
      refs = buildLiveCard(model);
      liveCards.set(model.id, refs);
      list.append(refs.card);
    }
    refs.card.querySelector('.url').textContent = model.url;
    if (refs.curl !== model.curl) {
      refs.curl = model.curl;
      refs.pre.textContent = model.curl;
    }
    const s = model.stats;
    const tps = s.last_tok_s ? `${s.last_tok_s.toFixed(1)} tok/s` : '—';
    refs.stats.textContent = `requests ${s.requests}  ·  completion tokens ${s.completion_tokens}  ·  last ${tps}`;
  }
  for (const [id, refs] of liveCards) {
    if (!liveIds.has(id)) {
      refs.card.remove();
      liveCards.delete(id);
    }
  }
}

$('rotate-key').addEventListener('click', async () => {
  if (!window.confirm('Create a new API key? Existing curl commands and apps will stop working until updated.')) return;
  try {
    await api('POST', '/api/serving/rotate-key');
  } catch (err) {
    window.alert(err.message);
  }
  refreshDashboard();
});

// ----- beam search ---------------------------------------------------------------------

let selectedJob = null;
const jobRows = new Map();

function parseWidths(text) {
  return text.split(',').map((v) => v.trim()).filter(Boolean).map(Number);
}

async function startBenchmark(auto) {
  const msg = $('beam-msg');
  msg.textContent = 'Starting…';
  msg.className = 'msg';
  const body = {
    model_id: $('beam-model').value,
    widths: parseWidths($('beam-widths').value),
    gen_tokens: Number($('beam-tokens').value),
    prompt: $('beam-prompt').value.trim() || null,
    auto,
  };
  try {
    const job = await api('POST', '/api/benchmarks', body);
    selectedJob = job.id;
    msg.textContent = `Started ${job.id}${auto ? ' (auto-optimize on)' : ''}`;
    refreshBenchmarks();
  } catch (err) {
    msg.textContent = err.message;
    msg.className = 'msg error';
  }
}

$('beam-form').addEventListener('submit', (e) => {
  e.preventDefault();
  startBenchmark(false);
});
$('beam-auto').addEventListener('click', () => startBenchmark(true));

function jobControls(job) {
  const controls = el('div', { class: 'btn-row' });
  const add = (label, run) => controls.append(el('button', {
    type: 'button',
    text: label,
    on: {
      click: async (ev) => {
        ev.stopPropagation();
        try {
          await run();
        } catch (err) {
          window.alert(err.message);
        }
        refreshBenchmarks();
      },
    },
  }));
  const post = (action, body) => () => api('POST', `/api/benchmarks/${job.id}/${action}`, body);
  const active = ['queued', 'running', 'pausing'].includes(job.status);
  if (job.status === 'running') add('PAUSE', post('pause'));
  if (job.status === 'paused') add('PLAY', post('resume'));
  if (job.auto) {
    add('STOP AUTO', post('auto', { enabled: false }));
  } else if (!active) {
    add('AUTO-OPTIMIZE', post('auto', { enabled: true }));
  }
  if (['queued', 'running', 'pausing', 'paused'].includes(job.status) || job.auto) {
    add('CANCEL', post('cancel'));
  }
  return controls;
}

function jobStatusText(job) {
  const auto = job.auto ? ' · AUTO' : '';
  if (job.status === 'running') return `RUNNING · width ${job.current_width ?? '…'}${auto}`;
  if (job.status === 'pausing') return `PAUSING after width ${job.current_width ?? '…'}${auto}`;
  return `${job.status.toUpperCase()}${auto}`;
}

async function refreshBenchmarks() {
  let data;
  try {
    data = await api('GET', '/api/benchmarks');
  } catch {
    return;
  }
  const body = $('jobs-body');
  const seen = new Set();
  const names = new Map(modelList.map((m) => [m.id, m.name]));
  for (const job of data.jobs) {
    seen.add(job.id);
    let row = jobRows.get(job.id);
    if (!row) {
      row = { tr: el('tr', { class: 'clickable', 'data-id': job.id }) };
      row.tr.addEventListener('click', () => {
        selectedJob = job.id;
        refreshBenchmarks();
      });
      jobRows.set(job.id, row);
      body.append(row.tr);
    }
    row.tr.className = `clickable${job.id === selectedJob ? ' selected' : ''}`;
    row.tr.replaceChildren(
      el('td', { class: 'mono', text: job.id }),
      el('td', { text: names.get(job.model_id) || job.model_id }),
      el('td', { class: 'mono', text: job.widths.join(', ') }),
      el('td', {},
        el('div', { class: `mono st st-${job.status === 'pausing' ? 'downloading' : job.status}`, text: jobStatusText(job) }),
        el('div', { class: 'sub', text: `${job.results.length}/${job.widths.length} widths done · ${formatWhen(job.updated_at)}` }),
        job.error ? el('div', { class: 'err-text', text: job.error }) : el('span'),
      ),
      el('td', {}, jobControls(job)),
    );
  }
  for (const [id, row] of jobRows) {
    if (!seen.has(id)) {
      row.tr.remove();
      jobRows.delete(id);
    }
  }
  if (!selectedJob && data.jobs.length) selectedJob = data.jobs[0].id;
  renderResults(data.jobs.find((j) => j.id === selectedJob));
}

function renderResults(job) {
  const body = $('results-body');
  if (!job) {
    body.replaceChildren();
    $('results-empty').hidden = false;
    $('results-title').textContent = 'Results';
    return;
  }
  $('results-empty').hidden = true;
  $('results-title').textContent = `Results · ${job.id}`;
  const lines = job.results.map((r) => el('tr', { class: r.best ? 'best' : '' },
    el('td', { class: 'mono', text: r.beam === 0 ? '0 (baseline)' : String(r.beam) }),
    el('td', { class: 'mono', text: r.search_s != null ? `${r.search_s.toFixed(1)} s` : '—' }),
    el('td', { class: 'mono', text: r.prefill_ms != null ? `${r.prefill_ms.toFixed(1)} ms` : '—' }),
    el('td', { class: 'mono', text: r.decode_ms_per_token != null ? r.decode_ms_per_token.toFixed(2) : '—' }),
    el('td', { class: 'mono', text: r.decode_tok_s != null ? r.decode_tok_s.toFixed(1) : '—' }),
    el('td', { class: 'mono', text: r.best ? '★ BEST' : '' }),
  ));
  if (job.status === 'running' || job.status === 'pausing') {
    lines.push(el('tr', {},
      el('td', { class: 'mono', text: `${job.current_width ?? '…'}` }),
      el('td', { class: 'sub', colspan: '5', text: 'measuring…' })));
  }
  body.replaceChildren(...lines);
}

// ----- kernels --------------------------------------------------------------------------

async function refreshKernels() {
  const modelId = $('kernel-model').value;
  if (!modelId) return;
  let kernelData;
  let benchData;
  try {
    [kernelData, benchData] = await Promise.all([
      api('GET', `/api/kernels?model_id=${encodeURIComponent(modelId)}`),
      api('GET', '/api/benchmarks'),
    ]);
  } catch (err) {
    $('best-run').textContent = err.message;
    return;
  }

  // Best run: lowest decode ms/token among all finished measurements for this model.
  let best = null;
  let bestJob = null;
  for (const job of benchData.jobs) {
    if (job.model_id !== modelId) continue;
    for (const r of job.results) {
      if (r.decode_ms_per_token == null) continue;
      if (!best || r.decode_ms_per_token < best.decode_ms_per_token) {
        best = r;
        bestJob = job.id;
      }
    }
  }
  const bestEl = $('best-run');
  if (!best) {
    bestEl.textContent = 'No benchmark results for this model yet.';
  } else {
    const kcount = kernelData.kernels.filter((k) => k.beam === best.beam && k.job_id === bestJob).length;
    bestEl.textContent =
      `Beam ${best.beam === 0 ? '0 (baseline)' : best.beam}: ${best.decode_ms_per_token.toFixed(2)} ms/token ` +
      `(${best.decode_tok_s.toFixed(1)} tok/s) · search ${best.search_s.toFixed(1)} s · ${kcount} kernels · job ${bestJob}`;
  }

  const kernels = kernelData.kernels;
  $('kernels-empty').hidden = kernels.length > 0;
  $('kernels-body').replaceChildren(...kernels.map((k) => el('tr', {},
    el('td', { class: 'mono', text: String(k.beam) }),
    el('td', { class: 'mono', text: k.ast_hash }),
    el('td', { class: 'mono opts', text: k.opts, title: k.opts }),
    el('td', { class: 'mono sub', text: k.job_id }),
  )));
}

$('kernel-model').addEventListener('change', refreshKernels);

// ----- scheduling -----------------------------------------------------------------------

let backendReady = false;
let tickCount = 0;

function refreshActiveTab() {
  if (!backendReady) return;
  if (activeTab === 'dashboard') refreshDashboard();
  if (activeTab === 'beam') refreshBenchmarks();
  if (activeTab === 'kernels') refreshKernels();
}

function startPolling() {
  refreshModels();
  refreshActiveTab();
  setInterval(() => {
    tickCount++;
    refreshModels();
    if (activeTab === 'kernels') {
      if (tickCount % 3 === 0) refreshKernels();
    } else {
      refreshActiveTab();
    }
  }, 1000);
}

// Poll until the backend is up, then load health info once.
async function init() {
  const poll = setInterval(async () => {
    const ready = await refreshStatus();
    if (ready) {
      clearInterval(poll);
      backendReady = true;
      loadHealth();
      startPolling();
    }
  }, 500);
  await refreshStatus();
}

init();
