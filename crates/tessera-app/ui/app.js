// Tessera — window logic.
//
// The engine does the work; this file only keeps the controls, the comparison
// and the progress readout in step with each other.

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const dialog = window.__TAURI__.dialog;

const IMAGE_FILTER = {
  name: 'Images',
  extensions: ['png', 'tif', 'tiff', 'jpg', 'jpeg', 'webp', 'bmp'],
};

const MIN_SCALE = 1;
const MAX_SCALE = 10;

const state = {
  file: null,
  scale: 4,
  style: 'maps',
  // Where in the image the comparison is looking, normalised.
  cx: 0.5,
  cy: 0.5,
  regionPx: 0,
  busy: false,
  // Guards against a slow preview landing after a newer one.
  previewToken: 0,
};

const el = (id) => document.getElementById(id);
const ui = {
  banner: el('banner'),
  bannerText: document.querySelector('.banner-text'),
  bannerClose: document.querySelector('.banner-close'),
  chip: el('filechip'),
  chipMark: document.querySelector('.filechip-mark'),
  chipName: document.querySelector('.filechip-name'),
  chipSub: document.querySelector('.filechip-sub'),
  scaleValue: el('scale-value'),
  scaleUp: el('scale-up'),
  scaleDown: el('scale-down'),
  stylePicker: el('style-picker'),
  resultDims: el('result-dims'),
  go: el('go'),
  stage: el('stage'),
  empty: el('empty'),
  browse: el('browse'),
  compare: el('compare'),
  viewport: el('viewport'),
  before: el('img-before'),
  after: el('img-after'),
  stampAfter: el('stamp-after'),
  divider: el('divider'),
  grip: el('grip'),
  navigator: el('navigator'),
  navImg: el('nav-img'),
  navBox: el('nav-box'),
  working: el('working'),
  workingTile: el('working-tile'),
  workingPct: el('working-pct'),
  meterFill: el('meter-fill'),
  spinner: el('spinner'),
  statusSize: el('status-size'),
  statusGeo: el('status-geo'),
  statusEngine: el('status-engine'),
  statusRight: el('status-right'),
};

// ---------------------------------------------------------------- messages

function showBanner(html, isError = false) {
  ui.bannerText.innerHTML = html;
  ui.banner.classList.toggle('is-error', isError);
  ui.banner.hidden = false;
}
ui.bannerClose.addEventListener('click', () => { ui.banner.hidden = true; });

const groupDigits = (n) => n.toLocaleString('en-US');

// ---------------------------------------------------------------- controls

function setScale(next) {
  state.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, next));
  ui.scaleValue.textContent = `${state.scale}×`;
  ui.scaleDown.disabled = state.scale === MIN_SCALE;
  ui.scaleUp.disabled = state.scale === MAX_SCALE;
  updateResultDims();
  schedulePreview();
}

function updateResultDims() {
  if (!state.file) {
    ui.resultDims.textContent = '—';
    return;
  }
  const w = state.file.width * state.scale;
  const h = state.file.height * state.scale;
  ui.resultDims.textContent = `${groupDigits(w)} × ${groupDigits(h)}`;
}

ui.scaleUp.addEventListener('click', () => setScale(state.scale + 1));
ui.scaleDown.addEventListener('click', () => setScale(state.scale - 1));

ui.stylePicker.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-style]');
  if (!button || button.classList.contains('on')) return;
  ui.stylePicker.querySelectorAll('button').forEach((b) => b.classList.remove('on'));
  button.classList.add('on');
  state.style = button.dataset.style;
  schedulePreview();
});

// ---------------------------------------------------------------- opening

async function pickFile() {
  if (state.busy) return;
  const picked = await dialog.open({ multiple: false, filters: [IMAGE_FILTER] });
  if (picked) loadImage(typeof picked === 'string' ? picked : picked.path);
}

ui.browse.addEventListener('click', pickFile);
ui.chip.addEventListener('click', pickFile);

async function loadImage(path) {
  setBusy(true, 'Reading image…');
  try {
    const info = await invoke('open_image', { path });
    state.file = info;
    state.cx = 0.5;
    state.cy = 0.5;

    ui.chip.classList.remove('is-empty');
    ui.chipName.textContent = info.name;
    ui.chipSub.textContent = `${groupDigits(info.width)} × ${groupDigits(info.height)}`;
    ui.chipMark.style.backgroundImage = `url("${info.preview}")`;
    ui.navImg.src = info.preview;

    ui.statusSize.textContent =
      `${groupDigits(info.width)} × ${groupDigits(info.height)}`;
    ui.statusGeo.hidden = !info.georeferenced;
    ui.statusRight.textContent = '';

    ui.empty.hidden = true;
    ui.compare.hidden = false;
    ui.go.disabled = false;

    updateResultDims();
  } catch (message) {
    showBanner(String(message), true);
    return;
  } finally {
    setBusy(false);
  }
  // Only once `busy` has cleared: `renderPreview` declines to run while the
  // window is busy, so calling it above would silently do nothing.
  renderPreview();
}

// ---------------------------------------------------------------- preview

let previewTimer = null;
function schedulePreview() {
  if (!state.file) return;
  clearTimeout(previewTimer);
  previewTimer = setTimeout(renderPreview, 260);
}

async function renderPreview() {
  if (!state.file || state.busy) return;
  const token = ++state.previewToken;
  ui.spinner.hidden = false;
  try {
    const detail = await invoke('preview_detail', {
      path: state.file.path,
      scale: state.scale,
      style: state.style,
      cx: state.cx,
      cy: state.cy,
    });
    // A newer request has already been issued; this result is stale.
    if (token !== state.previewToken) return;

    ui.before.src = detail.before;
    ui.after.src = detail.after;
    ui.after.decode().then(
      () => document.documentElement.style.setProperty(
        '--aspect', `${ui.after.naturalWidth} / ${ui.after.naturalHeight}`),
      () => {},
    );
    ui.stampAfter.textContent = `ENLARGED ${state.scale}×`;
    state.regionPx = detail.sourcePx;
    ui.statusEngine.textContent = detail.neural
      ? detail.engine.toUpperCase()
      : 'BASIC ENLARGEMENT — NO MODEL INSTALLED';
    drawNavBox();
  } catch (message) {
    if (token === state.previewToken) showBanner(String(message), true);
  } finally {
    if (token === state.previewToken) ui.spinner.hidden = true;
  }
}

// ---------------------------------------------------------------- comparing

function setSplit(fraction) {
  const pct = Math.min(100, Math.max(0, fraction * 100));
  document.documentElement.style.setProperty('--split', `${pct}%`);
}

function dragSplit(event) {
  const rect = ui.viewport.getBoundingClientRect();
  setSplit((event.clientX - rect.left) / rect.width);
}

ui.compare.addEventListener('pointerdown', (event) => {
  // The navigator sits on top of the comparison and has its own job.
  if (event.target.closest('#navigator')) return;
  ui.compare.setPointerCapture(event.pointerId);
  dragSplit(event);
});
ui.compare.addEventListener('pointermove', (event) => {
  if (event.buttons === 1 && !event.target.closest('#navigator')) dragSplit(event);
});

// ---------------------------------------------------------------- navigator

function drawNavBox() {
  if (!state.file || !state.regionPx) return;
  const fw = state.regionPx / state.file.width;
  const fh = state.regionPx / state.file.height;
  // Clamp the centre the same way the engine does, so the box tells the truth.
  const cx = Math.min(1 - fw / 2, Math.max(fw / 2, state.cx));
  const cy = Math.min(1 - fh / 2, Math.max(fh / 2, state.cy));
  ui.navBox.style.left = `${(cx - fw / 2) * 100}%`;
  ui.navBox.style.top = `${(cy - fh / 2) * 100}%`;
  ui.navBox.style.width = `${Math.min(100, fw * 100)}%`;
  ui.navBox.style.height = `${Math.min(100, fh * 100)}%`;
}

function moveRegion(event) {
  const rect = ui.navigator.getBoundingClientRect();
  state.cx = (event.clientX - rect.left) / rect.width;
  state.cy = (event.clientY - rect.top) / rect.height;
  drawNavBox();
  schedulePreview();
}

ui.navigator.addEventListener('pointerdown', (event) => {
  event.stopPropagation();
  ui.navigator.setPointerCapture(event.pointerId);
  moveRegion(event);
});
ui.navigator.addEventListener('pointermove', (event) => {
  if (event.buttons === 1) { event.stopPropagation(); moveRegion(event); }
});

// ---------------------------------------------------------------- running

function setBusy(busy, label) {
  state.busy = busy;
  ui.go.disabled = busy || !state.file;
  ui.scaleUp.disabled = busy || state.scale === MAX_SCALE;
  ui.scaleDown.disabled = busy || state.scale === MIN_SCALE;
  ui.stylePicker.querySelectorAll('button').forEach((b) => { b.disabled = busy; });
  if (label) {
    ui.spinner.hidden = false;
    ui.spinner.querySelector('span').textContent = label;
  } else if (!busy) {
    ui.spinner.hidden = true;
  }
}

function suggestedName() {
  const dot = state.file.name.lastIndexOf('.');
  const stem = dot > 0 ? state.file.name.slice(0, dot) : state.file.name;
  return `${stem}-${state.scale}x.png`;
}

ui.go.addEventListener('click', async () => {
  if (!state.file || state.busy) return;

  const output = await dialog.save({
    defaultPath: suggestedName(),
    filters: [{ name: 'PNG image', extensions: ['png'] },
              { name: 'TIFF image', extensions: ['tif', 'tiff'] }],
  });
  if (!output) return;

  setBusy(true);
  ui.spinner.hidden = true;
  ui.working.hidden = false;
  // The progress strip runs the full width along the bottom, where the
  // navigator lives; it is no use mid-run anyway.
  ui.navigator.hidden = true;
  ui.meterFill.style.width = '0%';
  ui.workingTile.textContent = 'Starting…';
  ui.workingPct.textContent = '0%';
  ui.statusRight.textContent = '';

  try {
    const result = await invoke('run_upscale', {
      path: state.file.path,
      output,
      scale: state.scale,
      style: state.style,
    });
    ui.statusRight.textContent = `FINISHED IN ${(result.elapsedMs / 1000).toFixed(1)}s`;
    ui.statusGeo.hidden = !result.georeferenced;
    showBanner(
      `Saved <code>${result.outputPath}</code> — ` +
      `${groupDigits(result.width)} × ${groupDigits(result.height)}, ${result.plan}.`
    );
  } catch (message) {
    showBanner(String(message), true);
  } finally {
    ui.working.hidden = true;
    ui.navigator.hidden = false;
    setBusy(false);
  }
});

listen('tessera:progress', (event) => {
  const { done, total } = event.payload;
  const pct = total ? Math.round((done / total) * 100) : 0;
  ui.meterFill.style.width = `${pct}%`;
  ui.workingPct.textContent = `${pct}%`;
  ui.workingTile.textContent = `TILE ${groupDigits(done)} / ${groupDigits(total)}`;
});

// ---------------------------------------------------------------- dropping

listen('tauri://drag-enter', () => ui.stage.classList.add('is-dragging'));
listen('tauri://drag-leave', () => ui.stage.classList.remove('is-dragging'));
listen('tauri://drag-drop', (event) => {
  ui.stage.classList.remove('is-dragging');
  const paths = event.payload && event.payload.paths;
  if (paths && paths.length) loadImage(paths[0]);
});

// ---------------------------------------------------------------- startup

(async function start() {
  setScale(state.scale);
  try {
    const status = await invoke('model_status');
    if (!status.models.length) {
      showBanner(
        'No enlarging model installed, so results will be plain resizing. ' +
        `Put a <code>.onnx</code> model in <code>${status.folder}</code> and restart.`
      );
    }
  } catch (message) {
    showBanner(String(message), true);
  }
})();
