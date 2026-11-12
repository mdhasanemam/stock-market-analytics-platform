/* Shared helpers: API client, formatting, theme, symbol autocomplete, Plotly theming. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts,
      body: opts.body && typeof opts.body !== 'string' ? JSON.stringify(opts.body) : opts.body });
  } catch (e) { throw new Error('Network error — is the server running?'); }
  let data = null;
  try { data = await res.json(); } catch (e) { /* empty body */ }
  if (!res.ok) {
    let msg = data && data.detail;
    if (Array.isArray(msg)) msg = msg.map((m) => m.msg).join('; ');
    throw new Error(msg || `Request failed (${res.status})`);
  }
  return data;
}

const fmt = {
  num: (v, d = 2) => (v == null || isNaN(v)) ? '—' : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }),
  pct: (v, d = 2) => (v == null || isNaN(v)) ? '—' : `${v > 0 ? '+' : ''}${Number(v).toFixed(d)}%`,
  ratioPct: (v, d = 1) => (v == null || isNaN(v)) ? '—' : `${v > 0 ? '+' : ''}${(v * 100).toFixed(d)}%`,
  vol: (v) => v == null ? '—' : v >= 1e9 ? (v / 1e9).toFixed(2) + 'B' : v >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(1) + 'K' : String(v),
  cls: (v) => (v > 0 ? 'up' : v < 0 ? 'down' : ''),
};

function toast(msg, kind = 'danger') {
  const el = document.createElement('div');
  el.className = `toast align-items-center text-bg-${kind} border-0`;
  el.setAttribute('role', 'alert');
  el.innerHTML = `<div class="d-flex"><div class="toast-body">${esc(msg)}</div><button class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button></div>`;
  $('#toasts').appendChild(el);
  const t = new bootstrap.Toast(el, { delay: 6000 }); t.show();
  el.addEventListener('hidden.bs.toast', () => el.remove());
}

function showError(container, msg) {
  container.innerHTML = `<div class="alert alert-danger d-flex align-items-start gap-2 mb-0" role="alert"><i class="bi bi-exclamation-triangle"></i><div>${esc(msg)}</div></div>`;
}
function showEmpty(container, icon, msg) {
  container.innerHTML = `<div class="empty"><i class="bi bi-${icon}"></i>${esc(msg)}</div>`;
}
function showLoading(container, lines = 1, h = 90) {
  container.innerHTML = Array.from({ length: lines }, () => `<div class="skeleton mb-2" style="min-height:${h}px"></div>`).join('');
}

/* ---- theme ---- */
function currentTheme() { return document.documentElement.getAttribute('data-bs-theme') || 'dark'; }
function applyThemeIcon() { const i = $('#themeToggle i'); if (i) i.className = currentTheme() === 'dark' ? 'bi bi-sun' : 'bi bi-moon-stars'; }
$('#themeToggle')?.addEventListener('click', () => {
  const n = currentTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-bs-theme', n);
  try { localStorage.setItem('theme', n); } catch (e) {}
  applyThemeIcon();
  document.dispatchEvent(new CustomEvent('themechange'));
});
applyThemeIcon();

/* ---- Plotly theming ---- */
function plotColors() {
  const dark = currentTheme() === 'dark';
  return { text: dark ? '#c9d1e0' : '#2b3445', grid: dark ? '#1e2738' : '#e3e8f0', up: '#16c784', down: '#ea3943',
           accent: dark ? '#2dd4a7' : '#0f9d78', blue: '#4c8dff', orange: '#ff9f43', grey: dark ? '#7b8699' : '#8a94a6', purple: '#a78bfa' };
}
function baseLayout(extra = {}) {
  const c = plotColors();
  return Object.assign({ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: c.text, size: 12 },
    margin: { l: 55, r: 20, t: 24, b: 40 }, hovermode: 'x unified', legend: { orientation: 'h', y: 1.08 },
    xaxis: { gridcolor: c.grid, rangeslider: { visible: false } }, yaxis: { gridcolor: c.grid } }, extra);
}
const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };

/* ---- symbol autocomplete (debounced) ---- */
function attachSymbolSearch(input, datalist) {
  let t;
  input.addEventListener('input', () => {
    clearTimeout(t);
    const q = input.value.trim();
    if (q.length < 1) return;
    t = setTimeout(async () => {
      try {
        const r = await api(`/api/stocks/search?q=${encodeURIComponent(q)}`);
        datalist.innerHTML = r.results.map((x) => `<option value="${esc(x.symbol)}">${esc(x.name)}</option>`).join('');
      } catch (e) { /* autocomplete is best-effort */ }
    }, 250);
  });
}
function parseSymbol(v) { return (v || '').trim().split(/[\s—-]/)[0].toUpperCase(); }
const nav = $('#navSearch');
if (nav) {
  attachSymbolSearch($('#navSymbol'), $('#navSymbols'));
  nav.addEventListener('submit', (e) => { e.preventDefault(); const s = parseSymbol($('#navSymbol').value); if (s) location.href = `/dashboard?symbol=${encodeURIComponent(s)}`; });
}
api('/api/health').then((h) => {
  const b = $('#providerBanner');
  if (h.data_provider === 'csv' && b) { b.classList.remove('d-none'); b.textContent = 'Demo mode: showing bundled historical data (ends Nov 2017), not current market prices.'; }
}).catch(() => {});
function qs(name, dflt = '') { return new URLSearchParams(location.search).get(name) || dflt; }
function sparkSvg(values, up) {
  const v = (values || []).filter((x) => x != null); if (v.length < 2) return '';
  const min = Math.min(...v), max = Math.max(...v), w = 120, h = 38, r = max - min || 1;
  const pts = v.map((x, i) => `${(i / (v.length - 1) * w).toFixed(1)},${(h - 3 - (x - min) / r * (h - 6)).toFixed(1)}`).join(' ');
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true"><polyline fill="none" stroke="${up ? '#16c784' : '#ea3943'}" stroke-width="1.8" points="${pts}"/></svg>`;
}
function stockCard(q, extra = '') {
  if (q.error) return `<div class="panel"><div class="sym fw-bold">${esc(q.symbol)}</div><div class="small text-danger">${esc(q.error)}</div></div>`;
  const up = q.change >= 0;
  return `<a class="panel stock-card" href="/dashboard?symbol=${encodeURIComponent(q.symbol)}">
    <div class="d-flex justify-content-between"><div><div class="sym">${esc(q.symbol)}</div><div class="nm text-truncate" style="max-width:11rem">${esc(q.name || '')}</div></div>${extra}</div>
    <div class="price mt-2">${fmt.num(q.close)}</div>
    <div class="${fmt.cls(q.change)} small fw-semibold">${up ? '▲' : '▼'} ${fmt.num(Math.abs(q.change))} (${fmt.pct(q.change_pct)})</div>
    ${sparkSvg(q.spark, up)}<div class="legend-note">as of ${esc(q.as_of)}</div></a>`;
}
