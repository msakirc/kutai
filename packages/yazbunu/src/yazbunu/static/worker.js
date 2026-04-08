"use strict";

// ─── Storage ───
let docs = [];      // parsed objects (null for malformed lines)
let rawLines = [];   // original strings

// Inverted index: field → lowercase-value → [lineIndex, ...]
let index = {};

// Fields to skip in inverted index (high cardinality / large)
const SKIP_INDEX = new Set(["ts", "exc"]);

// ─── Helpers ───

function addToIndex(fieldName, value, lineIdx) {
  if (SKIP_INDEX.has(fieldName)) return;
  const strVal = String(value).toLowerCase();
  if (!index[fieldName]) index[fieldName] = {};
  const bucket = index[fieldName];
  if (!bucket[strVal]) bucket[strVal] = [];
  bucket[strVal].push(lineIdx);
}

function parseDoc(line) {
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

// ─── Load ───

function handleLoad(lines, chunk) {
  const startIdx = docs.length;
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    rawLines.push(raw);
    const doc = parseDoc(raw);
    docs.push(doc);
    if (doc) {
      const idx = startIdx + i;
      for (const [k, v] of Object.entries(doc)) {
        if (v != null && v !== "") {
          addToIndex(k, v, idx);
        }
      }
    }
  }
  postMessage({ type: "loaded", count: lines.length, totalCount: docs.length, chunk });
}

// ─── Clear ───

function handleClear() {
  docs = [];
  rawLines = [];
  index = {};
}

// ─── Query Language Parser ───

function tokenize(query) {
  const tokens = [];
  let i = 0;
  const s = query;
  while (i < s.length) {
    // skip whitespace
    if (s[i] === " " || s[i] === "\t") { i++; continue; }
    // parentheses
    if (s[i] === "(") { tokens.push({ type: "LPAREN" }); i++; continue; }
    if (s[i] === ")") { tokens.push({ type: "RPAREN" }); i++; continue; }
    // quoted string after field:
    // check for operators AND, OR, NOT
    const upper = s.slice(i).toUpperCase();
    if (upper.startsWith("AND ") || upper.startsWith("AND\t") || upper.startsWith("AND)")) {
      tokens.push({ type: "AND" }); i += 3; continue;
    }
    if (upper.startsWith("OR ") || upper.startsWith("OR\t") || upper.startsWith("OR)")) {
      tokens.push({ type: "OR" }); i += 2; continue;
    }
    if (upper.startsWith("NOT ") || upper.startsWith("NOT\t")) {
      tokens.push({ type: "NOT" }); i += 3; continue;
    }
    // field:value or bare word
    let word = "";
    // read until whitespace or paren
    while (i < s.length && s[i] !== " " && s[i] !== "\t" && s[i] !== "(" && s[i] !== ")") {
      if (s[i] === '"') {
        // read quoted
        i++; // skip opening quote
        while (i < s.length && s[i] !== '"') {
          word += s[i]; i++;
        }
        if (i < s.length) i++; // skip closing quote
      } else if (s[i] === "'") {
        i++;
        while (i < s.length && s[i] !== "'") {
          word += s[i]; i++;
        }
        if (i < s.length) i++;
      } else {
        word += s[i]; i++;
      }
    }
    if (word) {
      // check if it's a field:value pair
      const colonIdx = word.indexOf(":");
      if (colonIdx > 0) {
        const field = word.slice(0, colonIdx);
        const value = word.slice(colonIdx + 1);
        tokens.push({ type: "TERM", field, value });
      } else {
        // bare word — match against msg
        tokens.push({ type: "TERM", field: "msg", value: word });
      }
    }
  }
  return tokens;
}

// Recursive descent parser for: expr = orExpr
// orExpr = andExpr (OR andExpr)*
// andExpr = notExpr ((AND)? notExpr)*
// notExpr = NOT? atom
// atom = LPAREN expr RPAREN | TERM

function parseQuery(query) {
  if (!query || !query.trim()) return null;
  const tokens = tokenize(query);
  if (tokens.length === 0) return null;
  let pos = 0;

  function peek() { return pos < tokens.length ? tokens[pos] : null; }
  function advance() { return tokens[pos++]; }

  function parseExpr() { return parseOr(); }

  function parseOr() {
    let left = parseAnd();
    while (peek() && peek().type === "OR") {
      advance(); // consume OR
      const right = parseAnd();
      left = { op: "OR", left, right };
    }
    return left;
  }

  function parseAnd() {
    let left = parseNot();
    while (peek()) {
      const t = peek();
      if (t.type === "AND") {
        advance();
        const right = parseNot();
        left = { op: "AND", left, right };
      } else if (t.type === "TERM" || t.type === "LPAREN" || t.type === "NOT") {
        // implicit AND
        const right = parseNot();
        left = { op: "AND", left, right };
      } else {
        break;
      }
    }
    return left;
  }

  function parseNot() {
    if (peek() && peek().type === "NOT") {
      advance();
      const operand = parseAtom();
      return { op: "NOT", operand };
    }
    return parseAtom();
  }

  function parseAtom() {
    const t = peek();
    if (!t) return { op: "TRUE" };
    if (t.type === "LPAREN") {
      advance();
      const expr = parseExpr();
      if (peek() && peek().type === "RPAREN") advance();
      return expr;
    }
    if (t.type === "TERM") {
      advance();
      return { op: "MATCH", field: t.field, value: t.value };
    }
    // unexpected token, skip
    advance();
    return { op: "TRUE" };
  }

  return parseExpr();
}

// ─── Time parsing ───

function parseRelativeTime(value) {
  // "5m ago", "2h ago", "1d ago"
  const m = value.match(/^(\d+)\s*(m|h|d)\s*ago$/i);
  if (!m) return null;
  const n = parseInt(m[1], 10);
  const unit = m[2].toLowerCase();
  const now = Date.now();
  const ms = unit === "m" ? n * 60000 : unit === "h" ? n * 3600000 : n * 86400000;
  return new Date(now - ms);
}

function parseTimeValue(value) {
  // try relative first
  const rel = parseRelativeTime(value);
  if (rel) return rel;
  // try ISO / date string
  const d = new Date(value);
  return isNaN(d.getTime()) ? null : d;
}

// ─── Evaluate query AST against a doc ───

function evalQuery(ast, doc) {
  if (!ast) return true;
  switch (ast.op) {
    case "TRUE": return true;
    case "AND": return evalQuery(ast.left, doc) && evalQuery(ast.right, doc);
    case "OR":  return evalQuery(ast.left, doc) || evalQuery(ast.right, doc);
    case "NOT": return !evalQuery(ast.operand, doc);
    case "MATCH": return evalMatch(ast.field, ast.value, doc);
    default: return true;
  }
}

function evalMatch(field, value, doc) {
  // Time-range filters
  if (field === "after") {
    const t = parseTimeValue(value);
    if (!t || !doc.ts) return false;
    return new Date(doc.ts) >= t;
  }
  if (field === "before") {
    const t = parseTimeValue(value);
    if (!t || !doc.ts) return false;
    return new Date(doc.ts) <= t;
  }
  if (field === "between") {
    // between:"start".."end"
    const parts = value.split("..");
    if (parts.length !== 2) return false;
    const t1 = parseTimeValue(parts[0]);
    const t2 = parseTimeValue(parts[1]);
    if (!t1 || !t2 || !doc.ts) return false;
    const dt = new Date(doc.ts);
    return dt >= t1 && dt <= t2;
  }

  // Numeric comparison
  if (value.startsWith(">") || value.startsWith("<")) {
    const op = value[0];
    const num = parseFloat(value.slice(1));
    if (isNaN(num)) return false;
    const docVal = parseFloat(doc[field]);
    if (isNaN(docVal)) return false;
    return op === ">" ? docVal > num : docVal < num;
  }

  // Substring match (case-insensitive)
  const docVal = doc[field];
  if (docVal == null) return false;
  return String(docVal).toLowerCase().includes(value.toLowerCase());
}

// ─── Filter ───

function handleFilter(query) {
  const ast = parseQuery(query);
  const indices = [];
  for (let i = 0; i < docs.length; i++) {
    const doc = docs[i];
    if (!doc) continue;
    if (evalQuery(ast, doc)) indices.push(i);
  }
  postMessage({ type: "filtered", indices, total: docs.length });
}

// ─── Stats ───

function handleStats() {
  const levels = {};
  const srcCounts = {};
  let minTs = null, maxTs = null;
  let count = 0;

  for (const doc of docs) {
    if (!doc) continue;
    count++;
    // Level distribution
    const lv = doc.level || "UNKNOWN";
    levels[lv] = (levels[lv] || 0) + 1;
    // Source counts
    if (doc.src) {
      srcCounts[doc.src] = (srcCounts[doc.src] || 0) + 1;
    }
    // Time range
    if (doc.ts) {
      if (!minTs || doc.ts < minTs) minTs = doc.ts;
      if (!maxTs || doc.ts > maxTs) maxTs = doc.ts;
    }
  }

  // Top 20 sources
  const sources = Object.entries(srcCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([name, count]) => ({ name, count }));

  postMessage({
    type: "stats",
    levels,
    sources,
    timeRange: { min: minTs, max: maxTs },
    count
  });
}

// ─── Aggregate ───

function truncateTs(ts, bucket) {
  const d = new Date(ts);
  if (isNaN(d.getTime())) return null;
  if (bucket === "minute") {
    return ts.slice(0, 16); // YYYY-MM-DDTHH:MM
  } else if (bucket === "hour") {
    return ts.slice(0, 13); // YYYY-MM-DDTHH
  } else { // day
    return ts.slice(0, 10); // YYYY-MM-DD
  }
}

function handleAggregate(field, bucket) {
  const buckets = {};
  for (const doc of docs) {
    if (!doc || !doc.ts) continue;
    const key = truncateTs(doc.ts, bucket);
    if (!key) continue;
    if (!buckets[key]) buckets[key] = { key, count: 0, levels: {} };
    buckets[key].count++;
    const lv = doc.level || "UNKNOWN";
    buckets[key].levels[lv] = (buckets[key].levels[lv] || 0) + 1;
  }
  const result = Object.values(buckets).sort((a, b) => a.key.localeCompare(b.key));
  postMessage({ type: "aggregated", buckets: result });
}

// ─── Pattern Detection ───

const UUID_RE = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;
const HEX_RE = /\b0x[0-9a-fA-F]+\b|(?<![a-zA-Z])[0-9a-fA-F]{8,}(?![a-zA-Z])/g;
const NUM_RE = /\b\d+(\.\d+)?\b/g;

function toTemplate(msg) {
  if (!msg) return "";
  let t = msg;
  t = t.replace(UUID_RE, "_UUID_");
  t = t.replace(HEX_RE, "_HEX_");
  t = t.replace(NUM_RE, "_N_");
  return t;
}

function handlePatterns() {
  const groups = {}; // template → { count, sample, indices, levels }
  for (let i = 0; i < docs.length; i++) {
    const doc = docs[i];
    if (!doc || !doc.msg) continue;
    const tmpl = toTemplate(doc.msg);
    if (!groups[tmpl]) {
      groups[tmpl] = { template: tmpl, count: 0, sample: doc.msg, indices: [], levels: {} };
    }
    const g = groups[tmpl];
    g.count++;
    if (g.indices.length < 5) g.indices.push(i);
    const lv = doc.level || "UNKNOWN";
    g.levels[lv] = (g.levels[lv] || 0) + 1;
  }

  const result = Object.values(groups)
    .sort((a, b) => b.count - a.count)
    .slice(0, 100);

  postMessage({ type: "patterns", groups: result });
}

// ─── Related Logs ───

function handleRelated(lineIndex, windowSec) {
  const doc = docs[lineIndex];
  if (!doc) { postMessage({ type: "related", indices: [] }); return; }

  const refTs = doc.ts ? new Date(doc.ts).getTime() : null;
  const windowMs = (windowSec || 60) * 1000;

  // Collect matching fields
  const matchFields = {};
  for (const key of ["task", "mission", "agent"]) {
    if (doc[key] != null && doc[key] !== "") matchFields[key] = doc[key];
  }
  if (Object.keys(matchFields).length === 0) {
    postMessage({ type: "related", indices: [] });
    return;
  }

  const indices = [];
  for (let i = 0; i < docs.length; i++) {
    if (i === lineIndex) continue;
    const d = docs[i];
    if (!d) continue;

    // Time window check
    if (refTs && d.ts) {
      const dt = new Date(d.ts).getTime();
      if (Math.abs(dt - refTs) > windowMs) continue;
    }

    // Check if any match field is shared
    let shared = false;
    for (const [k, v] of Object.entries(matchFields)) {
      if (d[k] != null && String(d[k]) === String(v)) { shared = true; break; }
    }
    if (shared) indices.push(i);
  }

  postMessage({ type: "related", indices });
}

// ─── Fields ───

function handleFields() {
  const names = Object.keys(index).sort();
  const topValues = {};
  for (const field of names) {
    const valMap = index[field];
    const entries = Object.entries(valMap)
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, 20);
    topValues[field] = entries.map(([val, idxArr]) => ({ value: val, count: idxArr.length }));
  }
  postMessage({ type: "fields", names, topValues });
}

// ─── Message handler ───

onmessage = function(e) {
  const msg = e.data;
  switch (msg.type) {
    case "load":
      handleLoad(msg.lines, msg.chunk);
      break;
    case "clear":
      handleClear();
      break;
    case "filter":
      handleFilter(msg.query);
      break;
    case "stats":
      handleStats();
      break;
    case "aggregate":
      handleAggregate(msg.field, msg.bucket);
      break;
    case "patterns":
      handlePatterns();
      break;
    case "related":
      handleRelated(msg.lineIndex, msg.windowSec);
      break;
    case "fields":
      handleFields();
      break;
  }
};
