// Test worker.js using Node vm module
import { readFileSync } from "fs";
import { createContext, runInContext } from "vm";

const workerCode = readFileSync(
  new URL("../src/yazbunu/static/worker.js", import.meta.url),
  "utf-8"
);

function createWorker() {
  const messages = [];
  const ctx = createContext({
    postMessage: (msg) => messages.push(msg),
    onmessage: null,
    console,
    Date,
    Object,
    Array,
    String,
    Math,
    JSON,
    Set,
    RegExp,
    isNaN,
    parseInt,
    parseFloat,
    NaN,
    Infinity,
  });
  runInContext(workerCode, ctx);
  return {
    send(data) {
      messages.length = 0;
      runInContext(`onmessage({data: ${JSON.stringify(data)}})`, ctx);
      return messages;
    },
  };
}

let passed = 0;
let failed = 0;
function assert(condition, name) {
  if (condition) {
    console.log(`  PASS: ${name}`);
    passed++;
  } else {
    console.error(`  FAIL: ${name}`);
    failed++;
  }
}

// Test data
const lines = [
  JSON.stringify({ ts: "2026-04-07T10:00:00.000Z", level: "INFO", src: "orchestrator", msg: "Task 42 started", task: 42, agent: "researcher" }),
  JSON.stringify({ ts: "2026-04-07T10:00:01.000Z", level: "ERROR", src: "orchestrator", msg: "Task 42 failed with error abc123def456", task: 42, duration_ms: 1500 }),
  JSON.stringify({ ts: "2026-04-07T10:00:02.000Z", level: "WARNING", src: "router", msg: "Model swap budget exceeded", model: "qwen" }),
  JSON.stringify({ ts: "2026-04-07T10:00:03.000Z", level: "DEBUG", src: "orchestrator", msg: "heartbeat ok" }),
  JSON.stringify({ ts: "2026-04-07T10:01:00.000Z", level: "INFO", src: "agents.base", msg: "Task 43 started", task: 43, agent: "researcher" }),
  "not valid json {{{",
  JSON.stringify({ ts: "2026-04-07T11:00:00.000Z", level: "INFO", src: "orchestrator", msg: "Processing item 550e8400-e29b-41d4-a716-446655440000" }),
];

const w = createWorker();

// ─── Test: Load ───
console.log("\n=== Load ===");
let res = w.send({ type: "load", lines, chunk: 0 });
assert(res.length === 1, "one response");
assert(res[0].type === "loaded", "type=loaded");
assert(res[0].count === 7, `count=${res[0].count} === 7`);
assert(res[0].totalCount === 7, `totalCount=${res[0].totalCount} === 7`);
assert(res[0].chunk === 0, "chunk=0");

// Load more
const moreLines = [
  JSON.stringify({ ts: "2026-04-07T12:00:00.000Z", level: "INFO", src: "router", msg: "Loaded model qwen" }),
];
res = w.send({ type: "load", lines: moreLines, chunk: 1 });
assert(res[0].count === 1, "incremental count=1");
assert(res[0].totalCount === 8, "total now 8");

// ─── Test: Stats ───
console.log("\n=== Stats ===");
res = w.send({ type: "stats" });
assert(res[0].type === "stats", "type=stats");
assert(res[0].levels.INFO === 4, `INFO=${res[0].levels.INFO} === 4`);
assert(res[0].levels.ERROR === 1, "ERROR=1");
assert(res[0].levels.WARNING === 1, "WARNING=1");
assert(res[0].levels.DEBUG === 1, "DEBUG=1");
assert(res[0].count === 7, `valid docs count=${res[0].count} === 7`);
assert(res[0].sources[0].name === "orchestrator", "top source=orchestrator");
assert(res[0].timeRange.min === "2026-04-07T10:00:00.000Z", "time min");
assert(res[0].timeRange.max === "2026-04-07T12:00:00.000Z", "time max");

// ─── Test: Filter — simple field match ───
console.log("\n=== Filter: level:error ===");
res = w.send({ type: "filter", query: "level:error" });
assert(res[0].type === "filtered", "type=filtered");
assert(res[0].indices.length === 1, `error count=${res[0].indices.length} === 1`);
assert(res[0].indices[0] === 1, "error at index 1");

// ─── Test: Filter — bare word ───
console.log("\n=== Filter: heartbeat (bare word) ===");
res = w.send({ type: "filter", query: "heartbeat" });
assert(res[0].indices.length === 1, "heartbeat matches 1 doc");

// ─── Test: Filter — AND + NOT ───
console.log("\n=== Filter: src:orchestrator AND NOT msg:heartbeat ===");
res = w.send({ type: "filter", query: 'src:orchestrator AND NOT msg:heartbeat' });
assert(res[0].indices.length === 3, `AND NOT count=${res[0].indices.length} === 3`);

// ─── Test: Filter — OR ───
console.log("\n=== Filter: level:error OR level:warning ===");
res = w.send({ type: "filter", query: "level:error OR level:warning" });
assert(res[0].indices.length === 2, `OR count=${res[0].indices.length} === 2`);

// ─── Test: Filter — numeric comparison ───
console.log("\n=== Filter: duration_ms:>1000 ===");
res = w.send({ type: "filter", query: "duration_ms:>1000" });
assert(res[0].indices.length === 1, `numeric match count=${res[0].indices.length} === 1`);
assert(res[0].indices[0] === 1, "matches the error line");

// ─── Test: Filter — parentheses ───
console.log("\n=== Filter: (task:42 OR task:43) AND agent:researcher ===");
res = w.send({ type: "filter", query: "(task:42 OR task:43) AND agent:researcher" });
assert(res[0].indices.length === 2, `paren group count=${res[0].indices.length} === 2`);

// ─── Test: Filter — empty query returns all ───
console.log("\n=== Filter: empty query ===");
res = w.send({ type: "filter", query: "" });
assert(res[0].indices.length === 7, `empty query returns all valid=${res[0].indices.length} === 7`);

// ─── Test: Aggregate ───
console.log("\n=== Aggregate ===");
res = w.send({ type: "aggregate", field: "ts", bucket: "hour" });
assert(res[0].type === "aggregated", "type=aggregated");
assert(res[0].buckets.length >= 2, `at least 2 hour buckets: ${res[0].buckets.length}`);
const firstBucket = res[0].buckets[0];
assert(firstBucket.key === "2026-04-07T10", "first bucket key");
assert(firstBucket.count >= 4, `first bucket count=${firstBucket.count} >= 4`);

// ─── Test: Patterns ───
console.log("\n=== Patterns ===");
res = w.send({ type: "patterns" });
assert(res[0].type === "patterns", "type=patterns");
assert(res[0].groups.length > 0, "has pattern groups");
// "Task 42 started" and "Task 43 started" should merge into same template
const taskPattern = res[0].groups.find(g => g.template.includes("Task _N_ started"));
assert(taskPattern != null, "Task _N_ started pattern found");
assert(taskPattern.count === 2, `task pattern count=${taskPattern ? taskPattern.count : 0} === 2`);
// UUID pattern
const uuidPattern = res[0].groups.find(g => g.template.includes("_UUID_"));
assert(uuidPattern != null, "UUID replacement pattern found");

// ─── Test: Related ───
console.log("\n=== Related ===");
res = w.send({ type: "related", lineIndex: 0, windowSec: 10 });
assert(res[0].type === "related", "type=related");
// line 0 has task:42, agent:researcher — line 1 shares task:42, line 4 shares agent:researcher but is 60s away
assert(res[0].indices.includes(1), "related includes line 1 (same task)");
assert(!res[0].indices.includes(0), "does not include self");

// ─── Test: Related — no match fields ───
console.log("\n=== Related: no match fields ===");
res = w.send({ type: "related", lineIndex: 6, windowSec: 60 });
assert(res[0].indices.length === 0, "no related when no task/mission/agent");

// ─── Test: Fields ───
console.log("\n=== Fields ===");
res = w.send({ type: "fields" });
assert(res[0].type === "fields", "type=fields");
assert(res[0].names.includes("level"), "has level field");
assert(res[0].names.includes("src"), "has src field");
assert(!res[0].names.includes("ts"), "ts excluded from index");
assert(!res[0].names.includes("exc"), "exc excluded from index");
assert(res[0].topValues.level.length > 0, "level has top values");

// ─── Test: Clear ───
console.log("\n=== Clear ===");
w.send({ type: "clear" });
res = w.send({ type: "stats" });
assert(res[0].count === 0, "clear resets count to 0");

// Summary
console.log(`\n${"=".repeat(40)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
