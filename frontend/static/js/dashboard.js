const state = { symbol: '', range: '1Y', start: '', end: '', hist: null, ind: null };
const PANELS = { volume: 'Volume', rsi: 'RSI (14)', macd: 'MACD', atr: 'ATR (14)', obv: 'OBV' };
attachSymbolSearch($('#symInput'), $('#symList'));

function rangeQuery() { return state.start || state.end ? `start=${state.start}&end=${state.end}` : `range=${state.range}`; }

async function load(symbol) {
  state.symbol = symbol;
  $('#symInput').value = symbol;
  history.replaceState(null, '', `/dashboard?symbol=${encodeURIComponent(symbol)}`);
  $('#toPredict').href = `/predict?symbol=${encodeURIComponent(symbol)}`;
  showLoading($('#statsRow'), 1, 70); $('#chart').innerHTML = '<div class="skeleton" style="min-height:520px"></div>';
  try {
    const [q, h, i] = await Promise.all([
      api(`/api/stocks/${symbol}/quote`),
      api(`/api/stocks/${symbol}/history?${rangeQuery()}`),
      api(`/api/stocks/${symbol}/indicators?${rangeQuery()}`)]);
    state.hist = h; state.ind = i;
    renderHeader(q, h); renderChart(); renderGuide(i.info);
  } catch (e) {
    $('#chart').innerHTML = ''; showError($('#statsRow'), e.message);
    $('#titleBlock').innerHTML = `<h1 class="h3 mb-0">${esc(symbol)}</h1>`;
  }
}

function renderHeader(q, h) {
  $('#titleBlock').innerHTML = `<h1 class="h3 mb-0">${esc(q.symbol)} <span class="text-secondary fs-6">${esc(q.name || '')}</span></h1>
    <div><span class="price">${fmt.num(q.close)}</span> <span class="${fmt.cls(q.change)} fw-semibold">${fmt.num(q.change)} (${fmt.pct(q.change_pct)})</span>
    <span class="text-secondary small ms-2">daily bar as of ${esc(q.as_of)}</span></div>`;
  const s = h.summary, cells = [['Open', fmt.num(q.open)], ['High', fmt.num(q.high)], ['Low', fmt.num(q.low)], ['Close', fmt.num(q.close)], ['Volume', fmt.vol(q.volume)],
    [`Change (${esc(s.start)} → ${esc(s.end)})`, `<span class="${fmt.cls(s.change_pct)}">${fmt.pct(s.change_pct)}</span>`], ['Range high', fmt.num(s.high)], ['Range low', fmt.num(s.low)]];
  $('#statsRow').innerHTML = cells.map(([k, v]) => `<div class="col-6 col-md-3 col-xl"><div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div></div>`).join('');
  const notes = [...(q.warnings || []), ...(h.warnings || [])];
  if (q.stale) notes.unshift('Live provider unavailable — showing cached data.');
  $('#dataNotes').innerHTML = [...new Set(notes)].map((n) => `<div class="caveat mb-2">${esc(n)}</div>`).join('');
}

function renderChart() {
  const c = plotColors(), h = state.hist, ind = state.ind;
  const overlays = $$('.ov:checked').map((x) => x.value), panels = $$('.pn:checked').map((x) => x.value);
  const traces = [], x = h.dates;
  if ($('#chartType').value === 'candle' && h.dates.length > 1)
    traces.push({ type: 'candlestick', x, open: h.open, high: h.high, low: h.low, close: h.close, name: state.symbol, increasing: { line: { color: c.up } }, decreasing: { line: { color: c.down } }, xaxis: 'x', yaxis: 'y' });
  else traces.push({ type: 'scatter', mode: 'lines', x, y: h.close, name: 'Close', line: { color: c.accent, width: 2 }, xaxis: 'x', yaxis: 'y' });
  const line = (name, y, color, extra = {}) => ({ type: 'scatter', mode: 'lines', x, y, name, line: { color, width: 1.4 }, xaxis: 'x', yaxis: 'y', ...extra });
  if (overlays.includes('sma20')) traces.push(line('SMA 20', ind.sma20, c.blue));
  if (overlays.includes('sma50')) traces.push(line('SMA 50', ind.sma50, c.orange));
  if (overlays.includes('ema20')) traces.push(line('EMA 20', ind.ema20, c.purple));
  if (overlays.includes('bollinger')) {
    traces.push(line('BB upper', ind.bollinger.upper, c.grey, { line: { color: c.grey, width: 1, dash: 'dot' } }));
    traces.push(line('BB lower', ind.bollinger.lower, c.grey, { line: { color: c.grey, width: 1, dash: 'dot' }, fill: 'tonexty', fillcolor: 'rgba(138,149,168,.10)' }));
  }
  const nP = panels.length, mainH = nP ? 0.56 : 1, gap = 0.04, subH = nP ? (1 - mainH - gap * nP) / nP : 0;
  const layout = baseLayout({ height: 480 + nP * 130, showlegend: true });
  layout.xaxis = { gridcolor: c.grid, rangeslider: { visible: false }, anchor: nP ? `y${nP + 1}` : 'y' };
  layout.yaxis = { gridcolor: c.grid, domain: [nP ? 1 - mainH : 0, 1], title: 'Price' };
  panels.forEach((p, k) => {
    const ax = k + 2, top = 1 - mainH - gap * (k + 1) - subH * k, dom = [top - subH, top], yname = `y${ax}`;
    layout[`yaxis${ax}`] = { gridcolor: c.grid, domain: dom, title: { text: PANELS[p], font: { size: 11 } } };
    const add = (t) => traces.push({ ...t, xaxis: 'x', yaxis: yname });
    if (p === 'volume') add({ type: 'bar', x, y: h.volume, name: 'Volume', marker: { color: h.close.map((v, i) => (i && v >= h.close[i - 1]) ? c.up : c.down), opacity: .65 } });
    if (p === 'rsi') { add({ type: 'scatter', mode: 'lines', x, y: ind.rsi, name: 'RSI', line: { color: c.accent, width: 1.5 } });
      layout.shapes = [...(layout.shapes || []), ...[30, 70].map((v) => ({ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: yname, y0: v, y1: v, line: { color: c.grey, width: 1, dash: 'dash' } }))]; layout[`yaxis${ax}`].range = [0, 100]; }
    if (p === 'macd') { add({ type: 'bar', x, y: ind.macd.hist, name: 'Histogram', marker: { color: ind.macd.hist.map((v) => v >= 0 ? c.up : c.down) }, opacity: .6 });
      add({ type: 'scatter', mode: 'lines', x, y: ind.macd.macd, name: 'MACD', line: { color: c.blue, width: 1.3 } });
      add({ type: 'scatter', mode: 'lines', x, y: ind.macd.signal, name: 'Signal', line: { color: c.orange, width: 1.3 } }); }
    if (p === 'atr') add({ type: 'scatter', mode: 'lines', x, y: ind.atr, name: 'ATR', line: { color: c.purple, width: 1.5 } });
    if (p === 'obv') add({ type: 'scatter', mode: 'lines', x, y: ind.obv, name: 'OBV', line: { color: c.blue, width: 1.5 } });
  });
  Plotly.react('chart', traces, layout, plotConfig);
}

function renderGuide(info) {
  $('#guide').innerHTML = Object.entries(info).map(([k, v], i) => `<div class="accordion-item bg-transparent"><h3 class="accordion-header">
    <button class="accordion-button collapsed bg-transparent" type="button" data-bs-toggle="collapse" data-bs-target="#g${i}">${esc(v.name)}</button></h3>
    <div id="g${i}" class="accordion-collapse collapse" data-bs-parent="#guide"><div class="accordion-body small text-secondary">${esc(v.description)}</div></div></div>`).join('')
    + '<p class="legend-note mt-3 mb-0">Indicators describe past prices. They are not signals or recommendations.</p>';
}

$('#symForm').addEventListener('submit', (e) => { e.preventDefault(); const s = parseSymbol($('#symInput').value); if (s) load(s); });
$$('#rangeBtns button').forEach((b) => b.addEventListener('click', () => {
  state.range = b.dataset.range; state.start = state.end = ''; $('#dStart').value = $('#dEnd').value = '';
  $$('#rangeBtns button').forEach((x) => x.classList.toggle('active', x === b)); if (state.symbol) load(state.symbol);
}));
$('#applyCustom').addEventListener('click', () => {
  const s = $('#dStart').value, e = $('#dEnd').value;
  if (!s && !e) return toast('Pick a start and/or end date.', 'warning');
  if (s && e && s > e) return toast('Start date must be on or before the end date.', 'warning');
  state.start = s; state.end = e; $$('#rangeBtns button').forEach((x) => x.classList.remove('active')); load(state.symbol);
});
$$('#toggles input, #chartType').forEach((el) => el.addEventListener('change', () => state.hist && renderChart()));
document.addEventListener('themechange', () => state.hist && renderChart());
$('#addWatch').addEventListener('click', async () => {
  if (!state.symbol) return toast('Load a stock first.', 'warning');
  try { await api('/api/watchlist', { method: 'POST', body: { symbol: state.symbol } }); toast(`${state.symbol} added to watchlist.`, 'success'); } catch (e) { toast(e.message); }
});
$(`#rangeBtns [data-range="${state.range}"]`).classList.add('active');
const initial = parseSymbol(qs('symbol'));
if (initial) load(initial); else showEmpty($('#chart'), 'search', 'Search for a ticker symbol or company name to see its chart.');
