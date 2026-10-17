// Universal Approximator — uniform x0 + step heights

const X_MIN = -25;
const X_MAX = 25;
const Y_MIN = -15;
const Y_MAX = 15;
const ASPECT = 5 / 3;
const MERGE_X_EPS = 0.05;
const CURVE_STEP = 0.15;
const GAME_PRECISION = 5;
const ACTIVE_ANCHOR_WEIGHT = 10;
const CLICK_LEFT_TOLERANCE = 0.08;
const VERTICAL_MAX_COEFF = 999;
const VERTICAL_MIN_EPS = 0.001;
const SIGMOID_K_MAX = 800;
const SIGMOID_K_VALUES = [
  ...Array.from({ length: 30 }, (_unused, index) => index + 1),
  40, 50, 60, 80, 100, 150, 200, 250, 300, SIGMOID_K_MAX,
];

const CANVAS_WIDTH = 1000;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;

const DEFAULT_PARAMS = {
  inputMode: "click",
  autoDetectActive: true,
  autoDetectEnemies: true,
  drawForwardOnly: false,
  approxMethod: "sigmoid",
  sampleStep: 0.5,
  sigmoidK: 100,
  numNeurons: 50,
  taylorOrder: 8,
  taylorHiddenLayers: 2,
  taylorHiddenSize: 16,
  fourierHarmonics: 12,
  fourierHiddenLayers: 0,
  fourierHiddenSize: 16,
  splineBoundary: "natural",
  splineUseBSpline: false,
  bsplineControlPoints: 12,
  bsplineSmoothing: 0,
  splinePlotStep: 0.15,
  splineFormulaPrecision: 14,
  mlpActivation: "tanh",
  trainEpochs: 500,
  trainLr: 0.02,
  animationEpochStep: 25,
  animationFrameMs: 80,
  showNeurons: true,
  stepHeights: true,
  freezeX0: true,
  dotPopulation: 48,
  dotControlPoints: 12,
  dotTrajectory: "spline",
  dotSplineSamples: 16,
  dotTargetRadius: 0.15,
  dotMutationScale: 0.9,
  dotEdgeOffset: 1.0,
  dotGenerationMs: 850,
  trajectoryRecordGenerations: 20,
  dotAvoidForbidden: true,
  dotShowForbidden: true,
};

const COLORS = {
  background: [28, 28, 32],
  gridMinor: [55, 55, 62, 80],
  gridMajor: [75, 75, 85, 120],
  axis: [140, 140, 150],
  axisLabel: [160, 160, 170],
  curve: [255, 90, 90],
  curveGlow: [255, 90, 90, 40],
  sample: [100, 180, 255],
  click: [255, 176, 80],
  clickGlow: [255, 176, 80, 50],
  anchor: [180, 140, 255],
  anchorGlow: [180, 140, 255, 55],
  dotAgent: [88, 185, 255],
  dotChampion: [80, 255, 140],
  dotEnemy: [255, 176, 80],
  dotUnreachable: [255, 90, 90],
  dotForbidden: [255, 55, 75],
  approx: [80, 255, 140],
  approxGlow: [80, 255, 140, 35],
  panelText: [210, 210, 220],
  neuronLine: [255, 200, 80, 90],
};

let params = { ...DEFAULT_PARAMS };
let clickPoints = [];
let clickWaypoints = [];
let drawnPoints = [];
let mergedWaypoints = [];
let linearWaypoints = [];
let trainingData = [];
let network = null;
let linearMse = null;
let sigmoidMse = null;
let taylorMse = null;
let fourierMse = null;
let splineMse = null;
let splineModel = null;
let approxXMin = null;
let approxXMax = null;
let isDrawing = false;
let controlsEl;
let statusMseEl;
let copyYBtnEl;
let copyY1BtnEl;
let copyY2BtnEl;
let copyBtnEl;
let captureBtnEl;
let recordTrainingBtnEl;
let recordTrajectoryBtnEl;
let trainingRecordingEnabled = false;
let bgImage = null;
let isCapturing = false;
let isTrainingAnimation = false;
let isTrajectoryRecording = false;
let dotPoints = [];
let dotEvolution = null;
let dotRunning = false;
let dotGenerationStartedAt = 0;
let forbiddenGrid = null;
let forbiddenStats = null;
let forbiddenError = null;
let activeAnchor = null;
let capturedActiveAnchor = null;
let capturedEnemyTargets = [];

function clonePoint(point) {
  return { x: Number(point.x), y: Number(point.y) };
}

function anchorPoint() {
  return activeAnchor ? clonePoint(activeAnchor.point) : null;
}

function seedAnchorPoints() {
  const point = anchorPoint();
  if (!point) return;
  clickPoints = [clonePoint(point)];
  dotPoints = [clonePoint(point)];
}

function setActiveAnchor(payload) {
  if (!payload || !payload.game) {
    activeAnchor = null;
    return;
  }
  activeAnchor = {
    point: {
      x: constrain(Number(payload.game.x), X_MIN, X_MAX),
      y: constrain(Number(payload.game.y), Y_MIN, Y_MAX),
    },
    pixel: payload.pixel || null,
    confidence: Number(payload.confidence ?? 0),
    uncertaintyGame: payload.uncertainty_game || { x: 0, y: 0 },
    method: payload.method || "unknown",
    needsReview: Boolean(payload.needs_review),
    source: "auto",
  };
  seedAnchorPoints();
}

function applyActiveDetectionPreference() {
  clearWorkspaceState({ keepAnchor: false });
  if (params.autoDetectActive && capturedActiveAnchor) {
    setActiveAnchor(capturedActiveAnchor);
  }
  updateStatusPanel();
  updateCopyButton();
}

function applyEnemyDetectionPreference() {
  if (params.inputMode === "dot") {
    dotPoints = activeAnchor ? [clonePoint(anchorPoint())] : [];
  } else if (params.inputMode === "click") {
    clickPoints = activeAnchor ? [clonePoint(anchorPoint())] : [];
  }
  if (params.autoDetectEnemies && capturedEnemyTargets.length) {
    const points = capturedEnemyTargets
      .map((target) => ({
        x: constrain(Number(target?.game?.x), X_MIN, X_MAX),
        y: constrain(Number(target?.game?.y), Y_MIN, Y_MAX),
      }))
      .sort((a, b) => a.x - b.x);
    if (params.inputMode === "dot") dotPoints.push(...points);
    else if (params.inputMode === "click") clickPoints.push(...points);
  }
  updateStatusPanel();
  updateCopyButton();
  updateDotButtons();
}

function anchorStatusText() {
  if (!activeAnchor) return "A: not detected";
  const uncertainty = Math.max(
    Number(activeAnchor.uncertaintyGame?.x || 0),
    Number(activeAnchor.uncertaintyGame?.y || 0)
  );
  const source = activeAnchor.source === "manual" ? "manual" : "auto";
  const review = activeAnchor.needsReview ? " — review A or disable auto-detection" : "";
  const confidence = roundCoord(Number(activeAnchor.confidence || 0));
  const point = activeAnchor.point;
  return `A ${source} (${roundCoord(point.x)},${roundCoord(point.y)}) conf=${confidence} ±${roundCoord(uncertainty)}${review}`;
}

function anchorScreenPosition() {
  if (!activeAnchor) return null;
  return worldToScreen(activeAnchor.point.x, activeAnchor.point.y);
}

function drawActiveAnchor() {
  if (!activeAnchor || params.inputMode !== "draw") return;
  const screen = anchorScreenPosition();
  const color = activeAnchor.needsReview ? COLORS.dotEnemy : COLORS.anchor;
  noStroke();
  fill(color[0], color[1], color[2], 55);
  circle(screen.x, screen.y, 28);
  fill(...color);
  circle(screen.x, screen.y, 13);
  fill(20, 20, 24);
  textAlign(CENTER, CENTER);
  textSize(11);
  text("A", screen.x, screen.y);
}

function setup() {
  const cnv = createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
  cnv.elt.oncontextmenu = () => false;
  document.getElementById("canvas-container").appendChild(cnv.elt);
  controlsEl = document.getElementById("controls-panel");
  buildControlsPanel();
  controlsEl.addEventListener("click", onHelpTriggerClick);
  controlsEl.addEventListener("keydown", onHelpTriggerKeydown);
  document.addEventListener("click", closeHelpPopups);
  loadServerOptions();
}

async function loadServerOptions() {
  try {
    const response = await fetch("/api/config");
    const options = await response.json();
    trainingRecordingEnabled = options.record_training_enabled === true;
  } catch {
    btn.textContent = "Error";
  }
  setTimeout(() => {
    if (btn) btn.textContent = prev;
  }, 1200);
}
  const btn = order === 1 ? copyY1BtnEl : copyY2BtnEl;
  if (!formula || !btn) return;
  const prev = btn.textContent;
  try {
    await navigator.clipboard.writeText(formula);
    btn.textContent = "Copied";
  } catch {
    alert(
      "Capture server unavailable.\nRun: python tools/approximator_server.py\nThen open http://127.0.0.1:8765/"
    );
  } finally {
    isCapturing = false;
    updateCaptureButton();
  }
}

function trainSigmoidNetwork() {
  const fitData = buildWeightedTrainingData(trainingData);
  network = new SigmoidNetwork(params.numNeurons, params.sigmoidK);
  network.initFromData(trainingData, approxXMin, approxXMax, {
    stepHeights: params.stepHeights,
  });
  network.train(
    fitData,
    approxXMin,
    approxXMax,
    params.trainEpochs,
    params.trainLr,
    params.freezeX0
  );
  network.mse = computeMseOnData((x) => network.predict(x), trainingData);
  sigmoidMse = network.mse;
}

function trainTaylorNetwork() {
  const fitData = buildWeightedTrainingData(trainingData);
  const activation = params.mlpActivation ?? "tanh";
  network = new TaylorNetwork(
    params.taylorOrder,
    params.taylorHiddenLayers,
    params.taylorHiddenSize,
    activation
  );
  network.initFromData(trainingData, approxXMin, approxXMax);
  network.train(fitData, params.trainEpochs, params.trainLr);
  network.mse = computeMseOnData((x) => network.predict(x), trainingData);
  taylorMse = network.mse;
}

function trainFourierNetwork() {
  const fitData = buildWeightedTrainingData(trainingData);
  const activation = params.mlpActivation ?? "tanh";
  network = new FourierNetwork(
    params.fourierHarmonics,
    params.fourierHiddenLayers,
    params.fourierHiddenSize,
    activation
  );
  network.initFromData(trainingData, approxXMin, approxXMax);
  network.train(fitData, params.trainEpochs, params.trainLr);
  network.mse = computeMseOnData((x) => network.predict(x), trainingData);
  fourierMse = network.mse;
}

function trainSplineModel() {
  network = null;
  splineModel = params.splineUseBSpline
    ? new BSplineModel(
        buildWeightedTrainingData(trainingData),
        params.bsplineControlPoints,
        params.bsplineSmoothing
      )
    : new CubicSplineModel(trainingData, params.splineBoundary);
  splineMse = computeMseOnData((x) => splineModel.predict(x), trainingData);
}

function processPipeline({ logFormula = false } = {}) {
  if (params.inputMode !== "draw" || drawnPoints.length === 0) return;

  trainingData = buildTrainingData(drawnPoints);
  if (trainingData.length < 2) {
    network = null;
    linearWaypoints = [];
    linearMse = null;
    sigmoidMse = null;
    taylorMse = null;
    fourierMse = null;
    splineMse = null;
    splineModel = null;
    approxXMin = null;
    approxXMax = null;
    if (statusMseEl) statusMseEl.textContent = "MSE: —";
    updateCopyButton();
    return;
  }

  syncLinearWaypointsFromDataset();
  approxXMin = trainingData[0].x;
  approxXMax = trainingData[trainingData.length - 1].x;
  linearMse = computeMseOnData((x) => evalLinearWaypoints(linearWaypoints, x), trainingData);

  if (params.approxMethod === "sigmoid") {
    trainSigmoidNetwork();
  } else if (params.approxMethod === "taylor") {
    trainTaylorNetwork();
  } else if (params.approxMethod === "fourier") {
    trainFourierNetwork();
  } else if (params.approxMethod === "spline") {
    trainSplineModel();
  } else {
    network = null;
    splineMse = null;
    splineModel = null;
  }

  updateStatusMse();
  updateCopyButton();
  if (logFormula) logActiveFormula();
}

function rerunPipeline() {
  readParamsFromUI();
  processPipeline();
}

function finishDrawing() {
  if (drawnPoints.length === 0) return;
  readParamsFromUI();
  processPipeline({ logFormula: true });
}

function formatMse(value) {
  return value !== null && Number.isFinite(value) ? roundCoord(value) : "—";
}

function roundCoord(value) {
  return Math.round(value * 1000) / 1000;
}

function worldToScreen(wx, wy) {
  return createVector(
    map(wx, X_MIN, X_MAX, 0, width),
    map(wy, Y_MAX, Y_MIN, 0, height)
  );
}

function screenToWorld(sx, sy) {
  return createVector(
    map(sx, 0, width, X_MIN, X_MAX),
    map(sy, 0, height, Y_MAX, Y_MIN)
  );
}

function lineWorld(x1, y1, x2, y2) {
  const a = worldToScreen(x1, y1);
  const b = worldToScreen(x2, y2);
  line(a.x, a.y, b.x, b.y);
}

function mousePressed() {
  if (!mouseInsideCanvas()) return;

  if (params.inputMode === "click") {
    if (mouseButton === RIGHT) {
      undoLastClick();
      return false;
    }
    if (mouseButton === LEFT) {
      addClickPointAtMouse();
    }
    return;
  }

  if (params.inputMode === "dot") {
    if (mouseButton === RIGHT) {
      undoLastDotPoint();
      return false;
    }
    if (mouseButton === LEFT) addDotPointAtMouse();
    return;
  }

  isDrawing = true;
  drawnPoints = [];
  mergedWaypoints = [];
  linearWaypoints = [];
  trainingData = [];
  network = null;
  linearMse = null;
  sigmoidMse = null;
  taylorMse = null;
  fourierMse = null;
  splineMse = null;
  splineModel = null;
  approxXMin = null;
  approxXMax = null;
  if (statusMseEl) statusMseEl.textContent = "MSE: —";
  updateCopyButton();
  addPointAtMouse();
}

function mouseDragged() {
  if (params.inputMode !== "draw" || !isDrawing || !mouseInsideCanvas()) return;
  addPointAtMouse();
}

function mouseReleased() {
  if (params.inputMode !== "draw" || !isDrawing) return;
  isDrawing = false;
  finishDrawing();
}

function addPointAtMouse() {
  const w = screenToWorld(mouseX, mouseY);
  w.x = constrain(w.x, X_MIN, X_MAX);
  w.y = constrain(w.y, Y_MIN, Y_MAX);
  if (params.drawForwardOnly && drawnPoints.length > 0) {
    const furthestX = drawnPoints.reduce((maxX, point) => Math.max(maxX, point.x), drawnPoints[0].x);
    w.x = Math.max(w.x, furthestX);
  }
  if (drawnPoints.length > 0) {
    const prev = drawnPoints[drawnPoints.length - 1];
    const dx = w.x - prev.x;
    const dy = w.y - prev.y;
    if (dx * dx + dy * dy < 0.02) return;
  }
  drawnPoints.push(w.copy());
}

function clampDrawnPointsToForwardX() {
  if (drawnPoints.length < 2) return;
  let furthestX = drawnPoints[0].x;
  for (let i = 1; i < drawnPoints.length; i++) {
    drawnPoints[i].x = Math.max(drawnPoints[i].x, furthestX);
    furthestX = drawnPoints[i].x;
  }
}

function isUndoShortcut() {
  if (!keyIsDown(CONTROL) && !keyIsDown(91)) return false;
  // EN: Ctrl+Z · RU keyboard: Ctrl+Я (same physical key, keyCode 90)
  if (keyCode === 90) return true;
  return key === "z" || key === "Z" || key === "я" || key === "Я";
}

function keyPressed() {
  if (params.inputMode === "click" && (keyCode === BACKSPACE || keyCode === DELETE || isUndoShortcut())) {
    undoLastClick();
    return false;
  }

  if (
    params.inputMode === "dot" &&
    (keyCode === BACKSPACE || keyCode === DELETE || isUndoShortcut())
  ) {
    undoLastDotPoint();
    return false;
  }

  if (params.inputMode === "dot" && key === " ") {
    if (dotRunning) {
      stopDotEvolution();
    } else {
      startDotEvolution();
    }
    return false;
  }

  if (key === "c" || key === "C" || key === "с" || key === "С") {
    if (params.inputMode === "click") {
      clearWorkspaceState();
      updateStatusPanel();
      return false;
    }
    if (params.inputMode === "dot") {
      clearDotPoints();
      return false;
    }
    drawnPoints = [];
    mergedWaypoints = [];
    linearWaypoints = [];
    trainingData = [];
    network = null;
    linearMse = null;
    sigmoidMse = null;
    taylorMse = null;
    fourierMse = null;
    splineMse = null;
    splineModel = null;
    approxXMin = null;
    approxXMax = null;
    if (statusMseEl) statusMseEl.textContent = "MSE: —";
    updateStatusPanel();
    updateCopyButton();
  }
}

function mouseInsideCanvas() {
  return mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height;
}
