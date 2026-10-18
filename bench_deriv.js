
// Comprehensive derivative performance benchmark for GraphBot
const math = require('mathjs');
global.window = { math };

// Stubs
function fmtGame(v) { return Math.round(v * 1e5) / 1e5; }
function roundCoord(v) { return Math.round(v * 1e4) / 1e4; }
function verticalEps(y1, y2) { return y1 !== y2 ? (y2 - y1) * 1e-6 : 1e-6; }
function normalizeFormula(text) {
  let s = String(text);
  let prev;
  do { prev = s;
    s = s.replace(/--/g, "+").replace(/\+\+/g, "+").replace(/\+-/g, "-").replace(/-\+/g, "-");
  } while (s !== prev);
  return s;
}
function round3(v) { if (!Number.isFinite(v)) return null; return Math.round(v * 1000) / 1000; }
function round4(v) { if (!Number.isFinite(v)) return null; return Math.round(v * 10000) / 10000; }

function derivativeCoeffs(coeffs, order) {
  const d = new Array(coeffs.length).fill(0);
  if (order <= 0) return coeffs.slice();
  for (let i = 0; i < coeffs.length; i++) {
    let val = coeffs[i];
    for (let k = 0; k < order; k++) val = val * (i - k);
    const idx = i - order;
    if (idx >= 0 && idx < d.length) d[idx] += val;
  }
  return d;
}

function bench(name, fn, iters = 1000) {
  for (let i = 0; i < Math.min(50, iters); i++) fn();
  const start = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) fn();
  const end = process.hrtime.bigint();
  const ms = Number(end - start) / 1e6;
  const usPerOp = (ms / iters * 1000);
  console.log(`${name}: ${ms.toFixed(2)} ms total, ${usPerOp.toFixed(3)} μs/op (${iters} iters)`);
  return { name, ms, usPerOp, iters };
}

const results = [];
results.push(bench("derivativeCoeffs order=1 (poly5)", () => derivativeCoeffs([1.5,-0.2,0.05,0.001,-0.0003], 1), 5000));
results.push(bench("derivativeCoeffs order=2 (poly5)", () => derivativeCoeffs([1.5,-0.2,0.05,0.001,-0.0003], 2), 5000));

// mathjs
const polyExpr = "1.5+-0.2*x+0.05*x^2+0.001*x^3+0.0002*x^4+-0.00003*x^5";
results.push(bench("mathjs order=1 (poly6)", () => {
  const node = math.parse(polyExpr);
  math.simplify(math.derivative(node, "x")).toString();
}, 200));
results.push(bench("mathjs order=2 (poly6)", () => {
  const node = math.parse(polyExpr);
  let d = math.derivative(node, "x");
  d = math.derivative(d, "x");
  math.simplify(d).toString();
}, 150));

// sigmoid 50 terms
const terms = [];
for (let i = 0; i < 50; i++) {
  const x0 = -25 + i * (50/49);
  const w = (Math.sin(i)*2).toFixed(3);
  terms.push(w + "/(1+exp(-100*(x-" + x0 + ")))" );
}
const sigExpr = "y=" + terms.join("+");
results.push(bench("mathjs order=1 (50-term sigmoid)", () => {
  const node = math.parse(sigExpr.substring(2));
  math.simplify(math.derivative(node, "x")).toString();
}, 50));
results.push(bench("mathjs order=2 (50-term sigmoid)", () => {
  const node = math.parse(sigExpr.substring(2));
  let d = math.derivative(node, "x");
  d = math.derivative(d, "x");
  math.simplify(d).toString();
}, 20));

console.log("\n---CACHE TEST---");
const cache = new Map();
const keys = [];
function cached(expr, compute) {
  if (cache.has(expr)) return cache.get(expr);
  const r = compute();
  if (cache.size >= 64) { const old = keys.shift(); if (old) cache.delete(old); }
  cache.set(expr, r); keys.push(expr);
  return r;
}
for (let i = 0; i < 50; i++) cached("x^2+3*x+sin(x)+exp(x)", () => {
  const node = math.parse("x^2+3*x+sin(x)+exp(x)");
  let d = math.derivative(node, "x");
  d = math.derivative(d, "x");
  return math.simplify(d).toString();
});
const start = process.hrtime.bigint();
for (let i = 0; i < 10000; i++) cached("x^2+3*x+sin(x)+exp(x)", () => {}); // won't recompute
const end = process.hrtime.bigint();
console.log("Cache hits (10k): " + (Number(end-start)/1e6).toFixed(3) + " ms");
console.log("\nJSON:" + JSON.stringify(results));
