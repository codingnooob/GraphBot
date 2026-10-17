// Cubic interpolation and cubic B-spline helpers for Draw mode.

class CubicSplineModel {
  constructor(data, boundary = "natural") {
    this.data = data.map((point) => ({ x: Number(point.x), y: Number(point.y) }));
    this.boundary = boundary;
    this.intervals = [];
    this.build();
  }

  build() {
    const n = this.data.length;
    if (n < 2) return;

    const h = [];
    const slopes = [];
    for (let i = 0; i < n - 1; i++) {
      const dx = this.data[i + 1].x - this.data[i].x;
      h.push(Math.max(Math.abs(dx), 1e-9));
      slopes.push((this.data[i + 1].y - this.data[i].y) / h[i]);
    }

    const matrix = Array.from({ length: n }, () => new Array(n).fill(0));
    const rhs = new Array(n).fill(0);
    if (this.boundary === "clamped") {
      matrix[0][0] = 2 * h[0];
      matrix[0][1] = h[0];
      rhs[0] = 6 * slopes[0];
      matrix[n - 1][n - 2] = h[n - 2];
      matrix[n - 1][n - 1] = 2 * h[n - 2];
      rhs[n - 1] = -6 * slopes[slopes.length - 1];
    } else {
      matrix[0][0] = 1;
      matrix[n - 1][n - 1] = 1;
    }

    for (let i = 1; i < n - 1; i++) {
      matrix[i][i - 1] = h[i - 1];
      matrix[i][i] = 2 * (h[i - 1] + h[i]);
      matrix[i][i + 1] = h[i];
      rhs[i] = 6 * (slopes[i] - slopes[i - 1]);
    }

    const second = solveLinearSystem(matrix, rhs);
    for (let i = 0; i < n - 1; i++) {
      const span = h[i];
      const a = this.data[i].y;
      const b = slopes[i] - (span * (2 * second[i] + second[i + 1])) / 6;
      const c = second[i] / 2;
      const d = (second[i + 1] - second[i]) / (6 * span);
      this.intervals.push({ x: this.data[i].x, a, b, c, d, h: span });
    }
  }

  predict(x) {
    if (this.intervals.length === 0) return 0;
    const index = intervalIndex(this.intervals, x);
    const part = x - this.intervals[index].x;
    const item = this.intervals[index];
    return item.a + part * (item.b + part * (item.c + part * item.d));
  }

  toFormulaText(decimals = 14) {
    return piecewiseCubicFormulaText(this.intervals, decimals);
  }

  toDerivativeFormulaText(order = 1, decimals = 14) {
    return piecewiseCubicDerivativeFormulaText(this.intervals, order, decimals);
  }
}

function piecewiseCubicFormulaText(intervals, decimals = 14) {
  if (!intervals || intervals.length === 0) return null;
  const terms = [];
  const first = intervals[0];
  const baseX = splineNumber(first.x, decimals);
  terms.push(splinePolynomial(first.a, first.b, first.c, first.d, `x - ${baseX}`, decimals));

  for (let i = 1; i < intervals.length; i++) {
    const jump = intervals[i].d - intervals[i - 1].d;
    if (Math.abs(jump) < 10 ** (-(decimals + 1))) continue;
    const knot = splineNumber(intervals[i].x, decimals);
    const positivePart = `((x - ${knot}) + abs(x - ${knot}))/2`;
    terms.push(`${splineNumber(jump, decimals)}*(${positivePart})^3`);
  }
  return `y=${normalizeSplineFormula(terms.join(" + "))}`;
}

function piecewiseCubicDerivativeFormulaText(intervals, order = 1, decimals = 14) {
  if (!intervals || intervals.length === 0) return null;
  const terms = [];
  const first = intervals[0];
  const baseX = splineNumber(first.x, decimals);
  if (order === 1) {
    const a1 = first.b;
    const b1 = 2 * first.c;
    const c1 = 3 * first.d;
    terms.push(splinePolynomial(a1, b1, c1, 0, `x - ${baseX}`, decimals));
    for (let i = 1; i < intervals.length; i++) {
      const prev = intervals[i - 1];
      const cur = intervals[i];
      const t = cur.x - prev.x;
      const y1_left = prev.b + 2 * prev.c * t + 3 * prev.d * t * t;
      const y1_right = cur.b;
      const jump = y1_right - y1_left;
      if (Math.abs(jump) < 10 ** (-(decimals + 1))) continue;
      const knot = splineNumber(cur.x, decimals);
      const positivePart = `((x - ${knot}) + abs(x - ${knot}))/2`;
      terms.push(`${splineNumber(jump, decimals)}*(${positivePart})`);
    }
    return terms.length > 0 ? `y'=${normalizeSplineFormula(terms.join(" + "))}` : null;
  } else if (order === 2) {
    const a2 = 2 * first.c;
    const b2 = 6 * first.d;
    terms.push(splinePolynomial(a2, b2, 0, 0, `x - ${baseX}`, decimals));
    for (let i = 1; i < intervals.length; i++) {
      const prev = intervals[i - 1];
      const cur = intervals[i];
      const t = cur.x - prev.x;
      const y2_left = 2 * prev.c + 6 * prev.d * t;
      const y2_right = 2 * cur.c;
      const jump = y2_right - y2_left;
      if (Math.abs(jump) < 10 ** (-(decimals + 1))) continue;
      const knot = splineNumber(cur.x, decimals);
      const positivePart = `((x - ${knot}) + abs(x - ${knot}))/2`;
      terms.push(`${splineNumber(jump, decimals)}*(${positivePart})`);
    }
    return terms.length > 0 ? `y''=${normalizeSplineFormula(terms.join(" + "))}` : null;
  }
  return null;

  toDerivativeFormulaText(order = 1, decimals = 14) {
    return piecewiseCubicDerivativeFormulaText(this.intervals, order, decimals);
  }
}

class BSplineModel {
  constructor(data, controlPointCount = 12, smoothing = 0) {
    this.data = data.map((point) => ({ x: Number(point.x), y: Number(point.y) }));
    this.controlPointCount = Math.max(4, Math.round(controlPointCount));
    this.smoothing = Math.max(0, Number(smoothing) || 0);
    this.degree = 3;
    this.knots = [];
    this.coefficients = [];
    this.intervals = [];
    this.build();
  }

  build() {
    if (this.data.length < 2) return;
    const xMin = this.data[0].x;
    const xMax = this.data[this.data.length - 1].x;
    const count = Math.min(this.controlPointCount, Math.max(4, this.data.length));
    this.controlPointCount = count;

    const internalCount = count - this.degree - 1;
    const spacing = (xMax - xMin) / Math.max(1, internalCount + 1);
    this.knots = [];
    for (let i = 0; i < this.degree + 1; i++) this.knots.push(xMin);
    for (let i = 1; i <= internalCount; i++) this.knots.push(xMin + i * spacing);
    for (let i = 0; i < this.degree + 1; i++) this.knots.push(xMax);

    const rows = this.data.map((point) =>
      Array.from({ length: count }, (_, index) =>
        bsplineBasis(index, this.degree, point.x, this.knots)
      )
    );
    const normal = Array.from({ length: count }, () => new Array(count).fill(0));
    const rhs = new Array(count).fill(0);
    for (let row = 0; row < rows.length; row++) {
      for (let i = 0; i < count; i++) {
        rhs[i] += rows[row][i] * this.data[row].y;
        for (let j = 0; j < count; j++) normal[i][j] += rows[row][i] * rows[row][j];
      }
    }

    const lambda = this.smoothing * 0.1 + 1e-8;
    for (let i = 0; i < count; i++) normal[i][i] += lambda;
    this.coefficients = solveLinearSystem(normal, rhs);
    this.intervals = this.buildPiecewiseIntervals();
  }

  predict(x) {
    if (this.coefficients.length === 0) return 0;
    let value = 0;
    for (let i = 0; i < this.coefficients.length; i++) {
      value += this.coefficients[i] * bsplineBasis(i, this.degree, x, this.knots);
    }
    return value;
  }

  buildPiecewiseIntervals() {
    const uniqueKnots = [];
    for (const knot of this.knots) {
      if (uniqueKnots.length === 0 || Math.abs(knot - uniqueKnots[uniqueKnots.length - 1]) > 1e-9) {
        uniqueKnots.push(knot);
      }
    }
    const intervals = [];
    const tValues = [0, 1 / 3, 2 / 3, 1];
    for (let i = 0; i < uniqueKnots.length - 1; i++) {
      const x0 = uniqueKnots[i];
      const h = uniqueKnots[i + 1] - x0;
      if (h <= 1e-9) continue;
      const values = tValues.map((t) => this.predict(x0 + t * h));
      const normalized = solveLinearSystem(
        tValues.map((t) => [1, t, t * t, t * t * t]),
        values
      );
      intervals.push({
        x: x0,
        a: normalized[0],
        b: normalized[1] / h,
        c: normalized[2] / (h * h),
        d: normalized[3] / (h * h * h),
        h,
      });
    }
    return intervals;
  }

  toFormulaText(decimals = 14) {
    return piecewiseCubicFormulaText(this.intervals, decimals);
  }

  toDerivativeFormulaText(order = 1, decimals = 14) {
    return piecewiseCubicDerivativeFormulaText(this.intervals, order, decimals);
  }

  toDerivativeFormulaText(order = 1, decimals = 14) {
    return piecewiseCubicDerivativeFormulaText(this.intervals, order, decimals);
  }
}function piecewiseCubicDerivativeFormulaText(intervals, order = 1, decimals = 14) {
  if (!intervals || intervals.length === 0) return null;
  const terms = [];
  const first = intervals[0];
  const baseX = splineNumber(first.x, decimals);
  if (order === 1) {
    const a1 = first.b;
    const b1 = 2 * first.c;
    const c1 = 3 * first.d;
    terms.push(splinePolynomial(a1, b1, c1, 0, `x - ${baseX}`, decimals));
    for (let i = 1; i < intervals.length; i++) {
      const prev = intervals[i - 1];
      const cur = intervals[i];
      const t = cur.x - prev.x;
      const y1_left = prev.b + 2 * prev.c * t + 3 * prev.d * t * t;
      const y1_right = cur.b;
      const jump = y1_right - y1_left;
      if (Math.abs(jump) < 10 ** (-(decimals + 1))) continue;
      const knot = splineNumber(cur.x, decimals);
      const positivePart = `((x - ${knot}) + abs(x - ${knot}))/2`;
      terms.push(`${splineNumber(jump, decimals)}*(${positivePart})`);
    }
    return terms.length > 0 ? `y'=${normalizeSplineFormula(terms.join(" + "))}` : null;
  } else if (order === 2) {
    const a2 = 2 * first.c;
    const b2 = 6 * first.d;
    terms.push(splinePolynomial(a2, b2, 0, 0, `x - ${baseX}`, decimals));
    for (let i = 1; i < intervals.length; i++) {
      const prev = intervals[i - 1];
      const cur = intervals[i];
      const t = cur.x - prev.x;
      const y2_left = 2 * prev.c + 6 * prev.d * t;
      const y2_right = 2 * cur.c;
      const jump = y2_right - y2_left;
      if (Math.abs(jump) < 10 ** (-(decimals + 1))) continue;
      const knot = splineNumber(cur.x, decimals);
      const positivePart = `((x - ${knot}) + abs(x - ${knot}))/2`;
      terms.push(`${splineNumber(jump, decimals)}*(${positivePart})`);
    }
    return terms.length > 0 ? `y''=${normalizeSplineFormula(terms.join(" + "))}` : null;
  }
  return null;
}

function piecewiseCubicFormulaText(intervals, decimals = 14) {
  if (!intervals || intervals.length === 0) return null;
  const terms = [];
  const first = intervals[0];
  const baseX = splineNumber(first.x, decimals);
  terms.push(splinePolynomial(first.a, first.b, first.c, first.d, `x - ${baseX}`, decimals));

  for (let i = 1; i < intervals.length; i++) {
    const jump = intervals[i].d - intervals[i - 1].d;
    if (Math.abs(jump) < 10 ** (-(decimals + 1))) continue;
    const knot = splineNumber(intervals[i].x, decimals);
    const positivePart = `((x - ${knot}) + abs(x - ${knot}))/2`;
    terms.push(`${splineNumber(jump, decimals)}*(${positivePart})^3`);
  }
  return `y=${normalizeSplineFormula(terms.join(" + "))}`;
}

function piecewiseCubicDerivativeFormulaText(intervals, order = 1, decimals = 14) {
  if (!intervals || intervals.length === 0) return null;
  const terms = [];
  const first = intervals[0];
  const baseX = splineNumber(first.x, decimals);
  if (order === 1) {
    const a1 = first.b;
    const b1 = 2 * first.c;
    const c1 = 3 * first.d;
    terms.push(splinePolynomial(a1, b1, c1, 0, `x - ${baseX}`, decimals));
    for (let i = 1; i < intervals.length; i++) {
      const prev = intervals[i - 1];
      const cur = intervals[i];
      const t = cur.x - prev.x;
      const y1_left = prev.b + 2 * prev.c * t + 3 * prev.d * t * t;
      const y1_right = cur.b;
      const jump = y1_right - y1_left;
      if (Math.abs(jump) < 10 ** (-(decimals + 1))) continue;
      const knot = splineNumber(cur.x, decimals);
      const positivePart = `((x - ${knot}) + abs(x - ${knot}))/2`;
      terms.push(`${splineNumber(jump, decimals)}*(${positivePart})`);
    }
    return terms.length > 0 ? `y'=${normalizeSplineFormula(terms.join(" + "))}` : null;
  } else if (order === 2) {
    const a2 = 2 * first.c;
    const b2 = 6 * first.d;
    terms.push(splinePolynomial(a2, b2, 0, 0, `x - ${baseX}`, decimals));
    for (let i = 1; i < intervals.length; i++) {
      const prev = intervals[i - 1];
      const cur = intervals[i];
      const t = cur.x - prev.x;
      const y2_left = 2 * prev.c + 6 * prev.d * t;
      const y2_right = 2 * cur.c;
      const jump = y2_right - y2_left;
      if (Math.abs(jump) < 10 ** (-(decimals + 1))) continue;
      const knot = splineNumber(cur.x, decimals);
      const positivePart = `((x - ${knot}) + abs(x - ${knot}))/2`;
      terms.push(`${splineNumber(jump, decimals)}*(${positivePart})`);
    }
    return terms.length > 0 ? `y''=${normalizeSplineFormula(terms.join(" + "))}` : null;
  }
  return null;
}
