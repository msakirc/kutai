"use strict";

const THEME_VARS = [
  "bg", "surface", "text", "text-dim", "border",
  "accent", "error", "warn", "info", "debug"
];

const BUILTIN_THEMES = {
  dark: { bg:"#1a1a2e", surface:"#16213e", text:"#e0e0e0", "text-dim":"#888", border:"#2a2a4a", accent:"#4fc3f7", error:"#ef5350", warn:"#ffa726", info:"#4fc3f7", debug:"#888" },
  light: { bg:"#f5f5f5", surface:"#ffffff", text:"#1a1a1a", "text-dim":"#666", border:"#ddd", accent:"#0277bd", error:"#c62828", warn:"#e65100", info:"#0277bd", debug:"#999" },
  monokai: { bg:"#272822", surface:"#1e1f1c", text:"#f8f8f2", "text-dim":"#75715e", border:"#3e3d32", accent:"#a6e22e", error:"#f92672", warn:"#fd971f", info:"#66d9ef", debug:"#75715e" },
  "solarized-dark": { bg:"#002b36", surface:"#073642", text:"#839496", "text-dim":"#586e75", border:"#073642", accent:"#268bd2", error:"#dc322f", warn:"#cb4b16", info:"#268bd2", debug:"#586e75" },
  "solarized-light": { bg:"#fdf6e3", surface:"#eee8d5", text:"#657b83", "text-dim":"#93a1a1", border:"#eee8d5", accent:"#268bd2", error:"#dc322f", warn:"#cb4b16", info:"#268bd2", debug:"#93a1a1" },
  nord: { bg:"#2e3440", surface:"#3b4252", text:"#eceff4", "text-dim":"#4c566a", border:"#434c5e", accent:"#88c0d0", error:"#bf616a", warn:"#d08770", info:"#88c0d0", debug:"#4c566a" },
  "high-contrast": { bg:"#000000", surface:"#111111", text:"#ffffff", "text-dim":"#aaaaaa", border:"#444444", accent:"#00ffff", error:"#ff0000", warn:"#ffff00", info:"#00ffff", debug:"#aaaaaa" },
  dracula: { bg:"#282a36", surface:"#21222c", text:"#f8f8f2", "text-dim":"#6272a4", border:"#44475a", accent:"#8be9fd", error:"#ff5555", warn:"#ffb86c", info:"#8be9fd", debug:"#6272a4" },
  "gruvbox-dark": { bg:"#282828", surface:"#1d2021", text:"#ebdbb2", "text-dim":"#928374", border:"#3c3836", accent:"#83a598", error:"#fb4934", warn:"#fe8019", info:"#83a598", debug:"#928374" },
  "catppuccin-mocha": { bg:"#1e1e2e", surface:"#181825", text:"#cdd6f4", "text-dim":"#585b70", border:"#313244", accent:"#89b4fa", error:"#f38ba8", warn:"#fab387", info:"#89b4fa", debug:"#585b70" },
  "tokyo-night": { bg:"#1a1b26", surface:"#16161e", text:"#c0caf5", "text-dim":"#565f89", border:"#292e42", accent:"#7aa2f7", error:"#f7768e", warn:"#e0af68", info:"#7aa2f7", debug:"#565f89" },
};

function applyThemeFromLib(name) {
  const theme = BUILTIN_THEMES[name] || loadCustomTheme(name);
  if (!theme) return;
  for (const [key, value] of Object.entries(theme))
    document.documentElement.style.setProperty("--" + key, value);
  localStorage.setItem("_yz_theme", name);
}

function loadCustomTheme(name) {
  try { return JSON.parse(localStorage.getItem("_yz_custom_themes") || "{}")[name] || null; } catch { return null; }
}

function saveCustomTheme(name, vars) {
  const c = JSON.parse(localStorage.getItem("_yz_custom_themes") || "{}");
  c[name] = vars;
  localStorage.setItem("_yz_custom_themes", JSON.stringify(c));
}

function getAllThemeNames() {
  const c = JSON.parse(localStorage.getItem("_yz_custom_themes") || "{}");
  return [...Object.keys(BUILTIN_THEMES), ...Object.keys(c)];
}

function getCurrentThemeVars() {
  const root = getComputedStyle(document.documentElement);
  const v = {};
  for (const k of THEME_VARS) v[k] = root.getPropertyValue("--" + k).trim();
  return v;
}

function exportTheme(name) {
  const t = BUILTIN_THEMES[name] || loadCustomTheme(name);
  return JSON.stringify({ name, vars: t }, null, 2);
}

function importTheme(json) {
  try { const { name, vars } = JSON.parse(json); saveCustomTheme(name, vars); return name; } catch { return null; }
}

function initTheme() {
  const saved = localStorage.getItem("_yz_theme");
  if (saved) { applyThemeFromLib(saved); return saved; }
  if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) { applyThemeFromLib("light"); return "light"; }
  applyThemeFromLib("dark");
  return "dark";
}
