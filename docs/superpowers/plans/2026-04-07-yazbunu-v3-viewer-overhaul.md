# Yazbunu v3 Viewer Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all viewer issues: lag from full DOM rebuild on every interaction, broken text selection, detail panel stealing screen space, and amateur visual quality.

**Architecture:** Single-file viewer (`viewer.html`) with inline CSS+JS. Zero deps, vanilla JS, no build tools. The server is dumb (file reader) — all intelligence stays in the browser. RAM is the bottleneck; browser does all heavy lifting.

**Tech Stack:** Vanilla JS, CSS custom properties, Web Workers (existing), aiohttp server (untouched)

**Key constraint from decisions doc:** No virtual scrolling. Flow-based DOM. Featherweight.

---

## File Map

All changes are in ONE file:
- **Modify:** `packages/yazbunu/src/yazbunu/static/viewer.html` (all CSS + JS inline)

No new files. No server changes. `worker.js`, `themes.js`, `server.py` stay untouched.

---

## Summary of Issues → Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| **Lag on every interaction** | `render()` rebuilds entire DOM; called from `selectRow()`, `vimMove()`, `hideDetail()`, bookmark toggle | New `updateSelection()` that only toggles CSS classes on 2 elements |
| **Text selection broken** | `.msg` has `white-space:nowrap; overflow:hidden; text-overflow:ellipsis` — text truncated. Also `render()` destroys DOM on click | Remove truncation from `.msg`. Stop rebuilding DOM on click |
| **Detail panel steals 400px** | Side panel with fixed 400px width, auto-opens on every click via `selectRow()→showDetail()` | Bottom drawer (not side panel), only opens on Enter/explicit action, resizable |
| **Looks amateur** | Flat styling, no visual hierarchy, per-row borders too noisy, toolbar is unstructured dump of controls | Toolbar grouping, remove per-row borders (use zebra + hover), subtle depth, refined spacing |

---

### Task 1: Fix the Render Loop (Performance)

**Files:**
- Modify: `packages/yazbunu/src/yazbunu/static/viewer.html` — JS section

The core fix. Currently 7 call sites trigger full DOM rebuild (`render()`) for actions that only need to change which row is highlighted. After this task, `render()` is only called for initial load, filter changes, and file switches.

- [ ] **Step 1: Add `updateSelection()` function**

Add this function after the existing `render()` function (around line 758). This replaces `render()` for selection-only changes:

```javascript
function updateSelection(oldVisibleIdx, newVisibleIdx) {
  // Remove old selection
  if (oldVisibleIdx >= 0 && oldVisibleIdx < $area.children.length) {
    const oldEl = $area.children[oldVisibleIdx];
    oldEl.classList.remove('selected');
    oldEl.removeAttribute('aria-selected');
  }
  // Add new selection
  if (newVisibleIdx >= 0 && newVisibleIdx < $area.children.length) {
    const newEl = $area.children[newVisibleIdx];
    newEl.classList.add('selected');
    newEl.setAttribute('aria-selected', 'true');
  }
}
```

- [ ] **Step 2: Fix `selectRow()` — stop calling `render()`**

Replace the current `selectRow` function:

```javascript
function selectRow(docIdx, visibleIdx) {
  const oldVisible = selectedRow;
  selectedDocIdx = docIdx;
  selectedRow = visibleIdx;
  updateSelection(oldVisible, visibleIdx);
}
```

Key change: no `render()`, no `showDetail()`. Selection is purely visual.

- [ ] **Step 3: Fix `vimMove()` — stop calling `render()`**

Replace the current `vimMove` function:

```javascript
function vimMove(delta) {
  const total = getVisibleCount();
  if (!total) return;
  const oldVisible = selectedRow;
  if (selectedRow < 0) selectedRow = 0;
  selectedRow = Math.max(0, Math.min(total - 1, selectedRow + delta));
  const docIdx = filteredIndices ? filteredIndices[selectedRow] : selectedRow;
  selectedDocIdx = docIdx;
  updateSelection(oldVisible, selectedRow);
  scrollToRow(selectedRow);
  // Update detail if drawer is open
  if ($detail.style.display !== 'none') showDetail(docIdx);
}
```

- [ ] **Step 4: Fix `hideDetail()` — stop calling `render()`**

Replace:
```javascript
function hideDetail() {
  $detail.style.display = 'none';
}
```

No more resetting `selectedDocIdx`/`selectedRow` or calling `render()`. The selection stays visible after closing the detail drawer.

- [ ] **Step 5: Fix `onRowClick()` — don't open detail on click**

Replace:
```javascript
function onRowClick(e) {
  if (e.target.classList.contains('pill')) return;
  const sel = window.getSelection();
  if (sel && sel.toString().length > 0) return;
  const row = e.currentTarget;
  const docIdx = parseInt(row.dataset.docIdx, 10);
  const visibleIdx = parseInt(row.dataset.visibleIdx, 10);
  selectRow(docIdx, visibleIdx);
}
```

Same logic but `selectRow` no longer triggers `render()` or `showDetail()`.

- [ ] **Step 6: Fix bookmark toggles — targeted update instead of full `render()`**

Replace `onRowDblClick`:
```javascript
function onRowDblClick(e) {
  const row = e.currentTarget;
  const docIdx = parseInt(row.dataset.docIdx, 10);
  const doc = allDocs[docIdx];
  if (!doc) return;
  const bk = bookmarkKey(doc);
  if (bookmarks.has(bk)) {
    bookmarks.delete(bk);
    row.classList.remove('bookmarked');
  } else {
    bookmarks.add(bk);
    row.classList.add('bookmarked');
  }
  saveBookmarks();
}
```

Replace `toggleBookmarkSelected`:
```javascript
function toggleBookmarkSelected(visibleIdx) {
  const total = getVisibleCount();
  if (visibleIdx < 0 || visibleIdx >= total) return;
  const el = $area.children[visibleIdx];
  if (!el) return;
  const docIdx = filteredIndices ? filteredIndices[visibleIdx] : visibleIdx;
  const doc = allDocs[docIdx];
  if (!doc) return;
  const bk = bookmarkKey(doc);
  if (bookmarks.has(bk)) {
    bookmarks.delete(bk);
    el.classList.remove('bookmarked');
  } else {
    bookmarks.add(bk);
    el.classList.add('bookmarked');
  }
  saveBookmarks();
}
```

Replace `toggleBookmarkIdx`:
```javascript
function toggleBookmarkIdx(docIdx) {
  const doc = allDocs[docIdx];
  if (!doc) return;
  const bk = bookmarkKey(doc);
  if (bookmarks.has(bk)) bookmarks.delete(bk);
  else bookmarks.add(bk);
  saveBookmarks();
  // Refresh detail panel button state
  showDetail(docIdx);
  // Update the row in DOM if visible
  const visibleIdx = filteredIndices ? filteredIndices.indexOf(docIdx) : docIdx;
  if (visibleIdx >= 0 && visibleIdx < $area.children.length) {
    const el = $area.children[visibleIdx];
    if (bookmarks.has(bk)) el.classList.add('bookmarked');
    else el.classList.remove('bookmarked');
  }
}
```

- [ ] **Step 7: Fix keyboard Enter handler — toggle detail explicitly**

In the keyboard handler `switch(e.key)`, the `Enter` case already toggles detail. Keep it but make sure it works with the new non-render flow. The existing logic is correct — it calls `showDetail()` or `hideDetail()` directly.

- [ ] **Step 8: Verify — open browser, press j/k rapidly**

Open `http://localhost:9880`, load a log file. Press j/k to navigate. Should be instant — no lag, no flicker. Selection highlight should move smoothly. Previously this triggered full DOM rebuild per keystroke.

- [ ] **Step 9: Commit**

```bash
git add packages/yazbunu/src/yazbunu/static/viewer.html
git commit -m "perf(yazbunu): targeted selection updates, stop rebuilding DOM on every interaction"
```

---

### Task 2: Fix Text Selection & Message Display

**Files:**
- Modify: `packages/yazbunu/src/yazbunu/static/viewer.html` — CSS section

- [ ] **Step 1: Fix `.msg` CSS — remove truncation**

Replace:
```css
.msg { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
```
With:
```css
.msg { flex: 1; min-width: 0; white-space: pre-wrap; word-break: break-word; }
```

`pre-wrap` preserves intentional whitespace in log messages while wrapping. `min-width: 0` prevents flex overflow. `word-break: break-word` handles long unbroken strings.

- [ ] **Step 2: Fix `.extras` — same truncation issue**

Replace:
```css
.extras { color: var(--text-dim); font-size: 11px; white-space: nowrap; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; max-width: 400px; }
```
With:
```css
.extras { color: var(--text-dim); font-size: 11px; flex-shrink: 0; }
```

Extra context fields should be fully visible, not hidden.

- [ ] **Step 3: Verify — select text across multiple log lines**

Open viewer, click and drag to select text across log lines. Text should be selectable and copyable. This was broken before because (a) truncation hid content, (b) render() destroyed DOM on click.

- [ ] **Step 4: Commit**

```bash
git add packages/yazbunu/src/yazbunu/static/viewer.html
git commit -m "fix(yazbunu): restore text selection, show full message content"
```

---

### Task 3: Detail Panel → Bottom Drawer

**Files:**
- Modify: `packages/yazbunu/src/yazbunu/static/viewer.html` — CSS + HTML + JS

The 400px side panel steals horizontal space from logs and auto-opens on every click. Replace with a bottom drawer that only opens explicitly and doesn't reduce log viewing area when collapsed.

- [ ] **Step 1: Replace side panel CSS with bottom drawer**

Replace the entire `.detail-panel` CSS block with:

```css
/* ─── Detail drawer (bottom) ─── */
.detail-panel {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  max-height: 40vh;
  background: var(--surface);
  border-top: 2px solid var(--accent);
  overflow-y: auto;
  padding: 12px 16px;
  box-shadow: 0 -4px 20px rgba(0,0,0,0.3);
  z-index: 20;
}
.detail-panel h3 {
  margin: 0 0 8px 0;
  font-size: 13px;
  color: var(--accent);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.detail-panel .close-btn {
  cursor: pointer;
  color: var(--text-dim);
  font-size: 16px;
  background: none;
  border: none;
  padding: 2px 6px;
  border-radius: 4px;
}
.detail-panel .close-btn:hover { color: var(--error); background: rgba(239,83,80,0.1); }
.detail-fields {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 4px 16px;
}
.detail-field {
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.detail-field .field-key {
  color: var(--accent);
  font-weight: bold;
}
.detail-field .field-val {
  color: var(--text);
  word-break: break-all;
}
.detail-exc {
  background: rgba(239,83,80,0.1);
  border-radius: 4px;
  padding: 8px;
  margin-top: 8px;
  font-size: 11px;
  white-space: pre-wrap;
  color: var(--error);
  max-height: 200px;
  overflow-y: auto;
}
.detail-related {
  margin-top: 8px;
  border-top: 1px solid var(--border);
  padding-top: 8px;
}
.detail-related .related-line {
  padding: 2px 0;
  font-size: 11px;
  cursor: pointer;
  color: var(--text-dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.detail-related .related-line:hover { color: var(--text); }
```

- [ ] **Step 2: Update `.main-content` to use relative positioning**

The `.main-content` stays as-is (flex:1, position:relative, overflow:hidden) — the drawer uses `position: absolute` within it. The log area still gets the full width.

- [ ] **Step 3: Update `showDetail()` — use grid layout for fields**

Replace `showDetail`:
```javascript
function showDetail(docIdx) {
  const doc = allDocs[docIdx];
  if (!doc) { hideDetail(); return; }

  $detail.style.display = 'block';
  let html = '<h3>Detail <button class="close-btn" onclick="hideDetail()" title="Escape">&times;</button></h3>';
  html += '<div class="detail-fields">';

  for (const [k, v] of Object.entries(doc)) {
    if (k === 'exc') continue;
    const escaped = String(v).replace(/</g, '&lt;').replace(/>/g, '&gt;');
    html += `<div class="detail-field"><span class="field-key">${k}:</span> <span class="field-val">${escaped}</span></div>`;
  }
  html += '</div>';

  const bk = bookmarkKey(doc);
  const isBookmarked = bookmarks.has(bk);
  html += `<div style="margin-top:8px"><button onclick="toggleBookmarkIdx(${docIdx})" style="font-size:11px;padding:4px 8px;background:var(--border);color:var(--text);border:none;border-radius:4px;cursor:pointer">${isBookmarked ? 'Unbookmark' : 'Bookmark'}</button></div>`;

  if (doc.exc) {
    html += `<div class="detail-exc">${doc.exc.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>`;
  }

  html += '<div class="detail-related" id="relatedArea"><em style="color:var(--text-dim);font-size:11px">Loading related...</em></div>';
  $detail.innerHTML = html;

  worker.postMessage({ type: 'related', lineIndex: docIdx, windowSec: 60 });
}
```

- [ ] **Step 4: Update mobile media query**

Replace the mobile `.detail-panel` override:
```css
@media (max-width: 600px) {
  .detail-panel { max-height: 50vh; }
}
```

The bottom drawer already works on mobile — just cap the height.

- [ ] **Step 5: Verify — open detail, logs still use full width**

Click a row, press Enter. Drawer appears at bottom. Log area above retains full width. Pressing Escape closes drawer. Logs are not pushed or resized.

- [ ] **Step 6: Commit**

```bash
git add packages/yazbunu/src/yazbunu/static/viewer.html
git commit -m "fix(yazbunu): detail panel as bottom drawer, stop stealing log width"
```

---

### Task 4: Visual Overhaul

**Files:**
- Modify: `packages/yazbunu/src/yazbunu/static/viewer.html` — CSS + HTML

Transform from flat amateur look to polished terminal-grade viewer. Changes are CSS-only where possible.

- [ ] **Step 1: Toolbar — grouped controls with visual separation**

Replace the toolbar CSS:

```css
.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 6px 12px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  align-items: center;
  z-index: 10;
}
.toolbar .group {
  display: flex;
  gap: 4px;
  align-items: center;
}
.toolbar .group + .group {
  margin-left: 2px;
  padding-left: 8px;
  border-left: 1px solid var(--border);
}
.toolbar select, .toolbar input, .toolbar button {
  font-family: inherit;
  font-size: 12px;
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 4px 8px;
  line-height: 1.4;
}
.toolbar button {
  cursor: pointer;
  font-weight: 600;
  transition: opacity 0.1s;
}
.toolbar button:hover { opacity: 0.85; }
.toolbar button.primary {
  background: var(--accent);
  color: #000;
  border-color: transparent;
}
.toolbar button.active {
  background: var(--error);
  color: #fff;
  border-color: transparent;
}
.toolbar input[type="text"] {
  flex: 1;
  min-width: 180px;
}
.toolbar select { max-width: 160px; }
.status {
  color: var(--text-dim);
  font-size: 11px;
  margin-left: auto;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
```

Update the toolbar HTML to use `<div class="group">` wrappers:

```html
<div class="toolbar" role="toolbar" aria-label="Log viewer controls">
  <div class="group">
    <select id="fileSelect" aria-label="Log file"><option value="">Loading...</option></select>
  </div>
  <div class="group">
    <select id="levelFilter" aria-label="Level filter">
      <option value="">All levels</option>
      <option value="DEBUG">DEBUG</option>
      <option value="INFO">INFO</option>
      <option value="WARNING">WARNING</option>
      <option value="ERROR">ERROR</option>
      <option value="CRITICAL">CRITICAL</option>
    </select>
    <select id="timeFilter" aria-label="Time range">
      <option value="">All time</option>
      <option value="1m">1m</option>
      <option value="5m" selected>5m</option>
      <option value="15m">15m</option>
      <option value="30m">30m</option>
      <option value="1h">1h</option>
      <option value="6h">6h</option>
      <option value="24h">24h</option>
    </select>
  </div>
  <div class="group" style="flex:1">
    <input type="text" id="searchBox" placeholder="level:ERROR src:db after:&quot;5m ago&quot;" aria-label="Search query">
  </div>
  <div class="group">
    <button id="tailBtn" class="primary" aria-label="Toggle live tail" aria-pressed="false">Tail</button>
    <div class="export-wrapper">
      <button id="exportBtn" aria-label="Export logs">Export</button>
      <div class="export-menu" id="exportMenu" role="menu">
        <button data-fmt="jsonl" role="menuitem">JSONL</button>
        <button data-fmt="json" role="menuitem">JSON</button>
        <button data-fmt="csv" role="menuitem">CSV</button>
        <button data-fmt="markdown" role="menuitem">Markdown</button>
        <button data-fmt="text" role="menuitem">Plain Text</button>
        <button data-fmt="clipboard" role="menuitem">Copy to Clipboard</button>
        <button data-fmt="share" role="menuitem">Share Link</button>
      </div>
    </div>
    <button id="notifyBtn" aria-label="Enable browser notifications">Notify</button>
    <select id="themeSelect" aria-label="Color theme"></select>
  </div>
  <span class="status" id="status" aria-live="polite">Ready</span>
</div>
```

- [ ] **Step 2: Log rows — remove noisy per-row borders, add zebra striping**

Replace log line base styles:

```css
.log-line {
  display: flex;
  align-items: baseline;
  padding: 1px 12px;
  gap: 8px;
  word-break: break-word;
  flex-wrap: wrap;
  user-select: text;
  border-left: 3px solid transparent;
}
.log-line:nth-child(even) { background: rgba(255,255,255,0.015); }
.log-line:hover { background: rgba(255,255,255,0.04); }
.log-line.selected { background: rgba(79,195,247,0.12); border-left-color: var(--accent); }
.log-line.level-ERROR { border-left-color: var(--error); }
.log-line.level-WARNING { border-left-color: var(--warn); }
.log-line.level-CRITICAL { border-left-color: var(--error); background: rgba(239,83,80,0.06); }
.log-line.level-DEBUG { opacity: 0.6; }
.log-line.selected.level-ERROR { background: rgba(239,83,80,0.18); border-left-color: var(--error); }
.log-line.selected.level-WARNING { background: rgba(255,167,38,0.18); border-left-color: var(--warn); }
.log-line.bookmarked { border-left-color: var(--warn); }
```

Key changes:
- Remove `border-bottom: 1px solid var(--border)` — per-row borders made it look like a spreadsheet
- Add `border-left: 3px solid transparent` — colored left border for level indication (like terminal severity)
- Zebra striping via `nth-child(even)` — subtle differentiation without borders
- `padding: 1px 12px` — tighter rows for more content density

- [ ] **Step 3: Field typography refinements**

```css
.ts { color: var(--text-dim); white-space: nowrap; font-size: 11px; flex-shrink: 0; font-variant-numeric: tabular-nums; }
.level { font-weight: 700; width: 44px; font-size: 10px; flex-shrink: 0; text-transform: uppercase; letter-spacing: 0.5px; }
.level-ERROR .level, .level-CRITICAL .level { color: var(--error); }
.level-WARNING .level { color: var(--warn); }
.level-INFO .level { color: var(--info); }
.level-DEBUG .level { color: var(--debug); }
.src { color: var(--accent); white-space: nowrap; flex-shrink: 0; max-width: 200px; overflow: hidden; text-overflow: ellipsis; opacity: 0.8; }
.msg { flex: 1; min-width: 0; white-space: pre-wrap; word-break: break-word; }
.extras { color: var(--text-dim); font-size: 11px; flex-shrink: 0; }
.fn-info { color: var(--text-dim); font-size: 10px; white-space: nowrap; flex-shrink: 0; opacity: 0.6; }
.pill {
  display: inline-block;
  font-size: 10px;
  padding: 0 5px;
  border-radius: 3px;
  background: var(--border);
  color: var(--text);
  cursor: pointer;
  margin-left: 4px;
  white-space: nowrap;
  flex-shrink: 0;
  line-height: 1.6;
}
.pill:hover { background: var(--accent); color: #000; }
```

- [ ] **Step 4: Body/root — tighter font, better base**

Add `--success` variable and refine body:
```css
:root {
  --bg: #1a1a2e;
  --surface: #16213e;
  --text: #e0e0e0;
  --text-dim: #888;
  --border: #2a2a4a;
  --accent: #4fc3f7;
  --error: #ef5350;
  --warn: #ffa726;
  --info: #4fc3f7;
  --debug: #888;
  --success: #66bb6a;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
  font-size: 12px;
  line-height: 1.5;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
}
```

Note: `font-size: 12px` (down from 13px) — tighter fits more lines on screen.

- [ ] **Step 5: Overlays — consistent backdrop blur**

Replace overlay styles:
```css
.cmd-palette-overlay, .shortcuts-overlay, .bookmarks-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.6);
  backdrop-filter: blur(2px);
  z-index: 200;
  display: none;
  justify-content: center;
}
.cmd-palette-overlay.open, .shortcuts-overlay.open, .bookmarks-overlay.open { display: flex; }
.cmd-palette-overlay { padding-top: 80px; align-items: flex-start; }
.shortcuts-overlay { align-items: center; }
.bookmarks-overlay { padding-top: 60px; align-items: flex-start; }
```

Panel shared styles:
```css
.cmd-palette, .shortcuts-panel, .bookmarks-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  overflow: hidden;
}
```

- [ ] **Step 6: Sparkline — reduce height, subtler**

```css
#sparkline {
  width: 100%;
  height: 24px;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  display: block;
}
```

Down from 32px. Uses `--bg` not `--surface` to blend with log area.

- [ ] **Step 7: Export menu — match new styling**

```css
.export-wrapper { position: relative; }
.export-menu {
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  z-index: 100;
  display: none;
  min-width: 140px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.export-menu.open { display: block; }
.export-menu button {
  display: block;
  width: 100%;
  text-align: left;
  background: none;
  color: var(--text);
  border: none;
  padding: 6px 12px;
  cursor: pointer;
  font-size: 12px;
}
.export-menu button:hover { background: var(--border); }
```

- [ ] **Step 8: Update sparkline height in JS `renderSparklineFromBuckets`**

In the `renderSparklineFromBuckets` function, change `const h = 32;` to `const h = 24;` and update `canvas.height = 24 * dpr;`. Same for `renderSparkline`.

- [ ] **Step 9: Verify visual result**

Open viewer. Check:
- Toolbar: controls grouped with separators
- Log rows: no per-row borders, subtle zebra, colored left border for severity
- Messages wrap naturally, fully selectable
- Detail drawer at bottom, not side
- Overlays have blur backdrop

- [ ] **Step 10: Commit**

```bash
git add packages/yazbunu/src/yazbunu/static/viewer.html
git commit -m "style(yazbunu): visual overhaul — grouped toolbar, density, severity borders, polished overlays"
```

---

### Task 5: Mobile & Edge Cases

**Files:**
- Modify: `packages/yazbunu/src/yazbunu/static/viewer.html` — CSS

- [ ] **Step 1: Update mobile media query**

```css
@media (max-width: 600px) {
  .toolbar { gap: 4px; padding: 6px 8px; }
  .toolbar .group { flex-wrap: wrap; width: 100%; }
  .toolbar .group + .group { margin-left: 0; padding-left: 0; border-left: none; border-top: 1px solid var(--border); padding-top: 4px; }
  .toolbar select, .toolbar input, .toolbar button { padding: 8px; font-size: 14px; }
  .log-line { font-size: 13px; padding: 2px 8px; }
  .ts { display: none; }
  .src { max-width: 60px; }
  .detail-panel { max-height: 50vh; }
  #minimap { display: none; }
  #sparkline { height: 20px; }
}
```

- [ ] **Step 2: Reduced motion — already correct**

Existing rule is fine:
```css
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
```

- [ ] **Step 3: Verify on narrow viewport**

Resize browser to <600px width. Toolbar should stack. Timestamps hide. Detail drawer still works at bottom.

- [ ] **Step 4: Commit**

```bash
git add packages/yazbunu/src/yazbunu/static/viewer.html
git commit -m "fix(yazbunu): mobile responsive layout for grouped toolbar"
```

---

### Task 6: Final Integration & Cleanup

**Files:**
- Modify: `packages/yazbunu/src/yazbunu/static/viewer.html`

- [ ] **Step 1: Remove stale CSS — `--row-height` variable**

Remove `--row-height: 28px;` from `:root` — it's a leftover from virtual scroll, not used anywhere in the new flow-based code.

- [ ] **Step 2: Verify the `$tailBtn` class uses `primary` not inline color**

In `startTail()`, the button gets class `active`. In `stopTail()`, it loses `active`. Update to use `primary` class when not tailing, `active` when tailing:

In `startTail`:
```javascript
$tailBtn.classList.remove('primary');
$tailBtn.classList.add('active');
```

In `stopTail`:
```javascript
$tailBtn.classList.remove('active');
$tailBtn.classList.add('primary');
```

`$notifyBtn` also needs cleanup — remove inline styles and use CSS classes:
```javascript
// In init, give it a class:
// Remove the inline style="background:var(--bg);color:var(--text);border:1px solid var(--border)" from HTML
// Add CSS class toggle in the notify handler
```

- [ ] **Step 3: Full end-to-end test**

Manual test checklist:
1. Load page — shows most recent logs, auto-scrolls to bottom, live tail starts
2. Press j/k — instant row navigation, no lag, no flicker
3. Select text with mouse — drag across rows, text stays selected
4. Press Enter on selected row — bottom drawer opens with details
5. Press Escape — drawer closes, selection persists
6. Change level filter — full re-render (expected), fast
7. Search for text — filter applies, results show
8. Click pill (task/mission) — breadcrumb added
9. Double-click row — bookmark toggle, no full re-render
10. Mobile width — toolbar stacks, no overflow

- [ ] **Step 4: Final commit**

```bash
git add packages/yazbunu/src/yazbunu/static/viewer.html
git commit -m "chore(yazbunu): cleanup stale CSS vars and button class management"
```

---

## What This Plan Does NOT Change

- **`worker.js`** — query engine, indexing, aggregation: all untouched
- **`themes.js`** — theme definitions: untouched (CSS variable approach still works)
- **`server.py`** — server API, auth, WebSocket: untouched
- **`sw.js`** — service worker: untouched
- **`manifest.json`** — PWA manifest: untouched

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| j/k keystroke | ~50-100ms (full DOM rebuild) | <1ms (class toggle) |
| Text selection | Broken (truncated + DOM destroyed) | Works naturally |
| Detail panel | 400px side panel, auto-opens | Bottom drawer, explicit open |
| Screen space for logs | 60-70% (panel open) | 100% (drawer overlays) |
| Visual quality | Flat, no hierarchy | Grouped toolbar, severity borders, density |
