# CELL
# Setup and imports
from __future__ import annotations

import os
import sys
import time
import json
import math
import random
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("psutil not installed - resource monitoring will be limited")

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("matplotlib/pandas/numpy not installed - charts will be skipped")

SEED = 42
random.seed(SEED)
np.random.seed(SEED) if HAS_PLOT else None

REPO_ROOT = Path("C:/Users/Tyron/GraphBot")
print(f"Repo root: {REPO_ROOT}")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")
print(f"CPU count: {os.cpu_count()}")
if HAS_PSUTIL:
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")

# CELL
# Check Node.js availability
try:
    result = subprocess.run(['node', '--version'], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=5)
    print(f"Node.js: {result.stdout.strip()}")
except Exception as e:
    print(f"Node.js not available: {e}")

# Check Python packages
required = ['cv2', 'numpy', 'psutil']
for pkg in required:
    try:
        __import__(pkg)
        print(f"âœ“ {pkg} installed")
    except ImportError:
        print(f"âœ— {pkg} NOT installed")

# Check mathjs availability
mathjs_path = REPO_ROOT / 'Visuals in p5.js' / 'universal-approximator' / 'https://cdnjs.cloudflare.com/ajax/libs/mathjs/11.11.2/math.min.js'
print(f"Frontend files exist:")
print(f"  sketch.js: {(REPO_ROOT / 'Visuals in p5.js/universal-approximator/sketch.js').exists()}")
print(f"  network.js: {(REPO_ROOT / 'Visuals in p5.js/universal-approximator/network.js').exists()}")
print(f"  spline.js: {(REPO_ROOT / 'Visuals in p5.js/universal-approximator/spline.js').exists()}")
print(f"  approximator_server.py: {(REPO_ROOT / 'tools/approximator_server.py').exists()}")

# CELL
# Create a comprehensive derivative benchmark script
benchmark_js = r'''
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
  console.log(`${name}: ${ms.toFixed(2)} ms total, ${usPerOp.toFixed(3)} Î¼s/op (${iters} iters)`);
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
'''

bench_file = REPO_ROOT / 'bench_deriv.js'
bench_file.write_text(benchmark_js)
print(f"Created: {bench_file}")

# Run benchmark
try:
    res = subprocess.run(['node', 'bench_deriv.js'], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
except Exception as e:
    print(f"Error: {e}")

# CELL
# Profile game_capture module
import cProfile
import pstats
from io import StringIO

print("Profiling game_capture imports and field_geometry...")
pr = cProfile.Profile()
pr.enable()
try:
    from core import field_geometry
    # test transforms
    for x in range(-25, 26, 5):
        for y in range(-15, 16, 5):
            px, py = field_geometry.game_to_pixel(x, y)
            gx, gy = field_geometry.pixel_to_game(px, py)
except Exception as e:
    print(f"Error: {e}")
pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(20)
print(s.getvalue())

# CELL
# Profile detection module
print("Profiling detection module...")
pr = cProfile.Profile()
pr.enable()
try:
    from core import detection, forbidden_mask
except Exception as e:
    print(f"Error: {e}")
pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(15)
print(s.getvalue())

# CELL
# Monitor resource usage during a short operation
if HAS_PSUTIL:
    proc = psutil.Process()
    print("Initial state:")
    print(f"  CPU %: {proc.cpu_percent(interval=0.1)}")
    print(f"  RSS: {proc.memory_info().rss / (1024**2):.2f} MB")
    print(f"  VMS: {proc.memory_info().vms / (1024**2):.2f} MB")
    
    # Simulate load: import cv2, numpy, etc.
    start = time.time()
    try:
        import cv2
        import numpy as np
        # create some arrays
        for _ in range(10):
            arr = np.random.rand(1000, 1000)
    except Exception as e:
        print(f"Error: {e}")
    elapsed = time.time() - start
    print(f"\nAfter load (elapsed {elapsed:.3f}s):")
    print(f"  CPU %: {proc.cpu_percent(interval=0.1)}")
    print(f"  RSS: {proc.memory_info().rss / (1024**2):.2f} MB")
else:
    print("psutil not available for detailed monitoring")

# CELL
# Run Python tests to confirm stability
try:
    res = subprocess.run([
        '.venv\Scripts\python.exe', '-m', 'unittest', 'discover', '-s', 'tests', '-v'
    ], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120)
    print(res.stdout[-500:] if len(res.stdout) > 500 else res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
except Exception as e:
    print(f"Error running tests: {e}")

