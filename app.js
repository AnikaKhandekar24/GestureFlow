const video = document.getElementById("webcam");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");

const startButton = document.getElementById("startButton");
const recordButton = document.getElementById("recordButton");
const saveButton = document.getElementById("saveButton");
const stopButton = document.getElementById("stopButton");
const gestureLabel = document.getElementById("gestureLabel");

const cameraStatus = document.getElementById("cameraStatus");
const gestureName = document.getElementById("gestureName");
const confidence = document.getElementById("confidence");
const actionName = document.getElementById("actionName");
const screenName = document.getElementById("screenName");
const recordPanel = document.getElementById("recordPanel");
const recordingState = document.getElementById("recordingState");
const sampleCount = document.getElementById("sampleCount");

const actions = {
  swipe_left: "Previous slide",
  swipe_right: "Next slide",
  swipe_up: "Scroll up",
  swipe_down: "Scroll down",
  open_palm: "Pause",
  fist: "Select",
  peace: "Take screenshot",
  unknown: "Show response"
};

const screens = ["Dashboard", "Gallery", "Slides", "Notes"];
let screenIndex = 0;
let camera = null;
let hands = null;
let isRunning = false;
let isRecording = false;
let samples = [];
let motionHistory = [];
let cooldown = 0;
let lastGesture = "Ready";
let lastAction = "Start the camera";
let lastConfidence = 0;
let activeFilter = 0;
let illusionTrails = [];
let illusionParticles = [];
let lastIllusionPoint = null;

const filters = [
  "none",
  "saturate(1.22)",
  "contrast(1.14) brightness(1.05)",
  "sepia(0.28) saturate(1.18)"
];

const illusionPoints = [0, 4, 8, 12, 16, 20];

function formatGesture(name) {
  return name
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function setStatus(text) {
  cameraStatus.textContent = text;
}

function updateUi(gesture, score, action) {
  lastGesture = gesture;
  lastConfidence = score;
  lastAction = action;
  gestureName.textContent = formatGesture(gesture);
  confidence.textContent = `Confidence: ${score.toFixed(2)}`;
  actionName.textContent = action;
  screenName.textContent = `Screen: ${screens[screenIndex]}`;
}

function landmarkRow(label, landmarks) {
  const values = [label];
  for (const point of landmarks) {
    values.push(point.x.toFixed(6), point.y.toFixed(6), point.z.toFixed(6));
  }
  return values;
}

function normalizedDistance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function fingerStates(landmarks) {
  const wrist = landmarks[0];
  const middleBase = landmarks[9];
  const palmSize = Math.max(normalizedDistance(wrist, middleBase), 0.001);
  const isRightHand = landmarks[5].x < landmarks[17].x;

  return {
    thumb: isRightHand ? landmarks[4].x > landmarks[3].x : landmarks[4].x < landmarks[3].x,
    index: normalizedDistance(wrist, landmarks[8]) > normalizedDistance(wrist, landmarks[6]) + palmSize * 0.12,
    middle: normalizedDistance(wrist, landmarks[12]) > normalizedDistance(wrist, landmarks[10]) + palmSize * 0.12,
    ring: normalizedDistance(wrist, landmarks[16]) > normalizedDistance(wrist, landmarks[14]) + palmSize * 0.12,
    pinky: normalizedDistance(wrist, landmarks[20]) > normalizedDistance(wrist, landmarks[18]) + palmSize * 0.12
  };
}

function classifyStatic(landmarks) {
  const fingers = fingerStates(landmarks);
  const up = Object.values(fingers).filter(Boolean).length;

  if (fingers.index && fingers.middle && !fingers.ring && !fingers.pinky) {
    return { name: "peace", confidence: fingers.thumb ? 0.78 : 0.88 };
  }

  if (up >= 4) {
    return { name: "open_palm", confidence: Math.min(0.72 + up * 0.05, 0.96) };
  }

  if (up <= 1) {
    return { name: "fist", confidence: 0.88 };
  }

  return { name: "unknown", confidence: 0.42 };
}

function classifyMotion(landmarks) {
  const wrist = landmarks[0];
  const indexTip = landmarks[8];
  motionHistory.push({
    x: (wrist.x + indexTip.x) / 2,
    y: (wrist.y + indexTip.y) / 2
  });

  if (motionHistory.length > 18) {
    motionHistory.shift();
  }

  if (motionHistory.length < 18) {
    return null;
  }

  const start = motionHistory[0];
  const end = motionHistory[motionHistory.length - 1];
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const threshold = 0.18;

  if (Math.abs(dx) < threshold && Math.abs(dy) < threshold) {
    return null;
  }

  if (Math.abs(dx) > Math.abs(dy)) {
    return {
      name: dx > 0 ? "swipe_right" : "swipe_left",
      confidence: Math.min(Math.abs(dx) / 0.44, 1)
    };
  }

  return {
    name: dy > 0 ? "swipe_down" : "swipe_up",
    confidence: Math.min(Math.abs(dy) / 0.44, 1)
  };
}

function landmarkToCanvas(point, width, height) {
  return {
    x: point.x * width,
    y: point.y * height
  };
}

function addMotionIllusions(landmarks, width, height) {
  const anchors = illusionPoints.map((index) => landmarkToCanvas(landmarks[index], width, height));
  const center = anchors.reduce(
    (total, point) => ({
      x: total.x + point.x / anchors.length,
      y: total.y + point.y / anchors.length
    }),
    { x: 0, y: 0 }
  );

  const speed = lastIllusionPoint
    ? Math.hypot(center.x - lastIllusionPoint.x, center.y - lastIllusionPoint.y)
    : 0;

  illusionTrails.push({
    points: anchors,
    speed,
    age: 0,
    life: 18
  });

  if (illusionTrails.length > 26) {
    illusionTrails.shift();
  }

  if (speed > 16) {
    for (const point of anchors) {
      illusionParticles.push({
        x: point.x,
        y: point.y,
        vx: (Math.random() - 0.5) * speed * 0.16,
        vy: (Math.random() - 0.5) * speed * 0.16,
        radius: 2 + Math.random() * 3,
        age: 0,
        life: 18 + Math.random() * 14,
        hue: Math.random() > 0.5 ? 138 : 190
      });
    }
  }

  if (illusionParticles.length > 190) {
    illusionParticles.splice(0, illusionParticles.length - 190);
  }

  lastIllusionPoint = center;
}

function drawMotionIllusions(width, height) {
  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  for (const trail of illusionTrails) {
    const opacity = Math.max(0, 1 - trail.age / trail.life);
    const intensity = Math.min(trail.speed / 36, 1);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.shadowBlur = 18 + intensity * 18;
    ctx.shadowColor = `rgba(54, 240, 111, ${0.32 * opacity})`;
    ctx.strokeStyle = `rgba(54, 240, 111, ${0.12 + 0.34 * opacity})`;
    ctx.lineWidth = 2 + intensity * 5;

    ctx.beginPath();
    ctx.moveTo(trail.points[0].x, trail.points[0].y);
    for (let index = 1; index < trail.points.length; index += 1) {
      const previous = trail.points[index - 1];
      const current = trail.points[index];
      const midX = (previous.x + current.x) / 2;
      const midY = (previous.y + current.y) / 2;
      ctx.quadraticCurveTo(previous.x, previous.y, midX, midY);
    }
    ctx.stroke();

    for (const point of trail.points) {
      const glow = 10 + intensity * 20;
      const gradient = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, glow);
      gradient.addColorStop(0, `rgba(255, 47, 85, ${0.44 * opacity})`);
      gradient.addColorStop(0.45, `rgba(54, 240, 111, ${0.2 * opacity})`);
      gradient.addColorStop(1, "rgba(54, 240, 111, 0)");
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(point.x, point.y, glow, 0, Math.PI * 2);
      ctx.fill();
    }

    trail.age += 1;
  }

  for (const particle of illusionParticles) {
    const opacity = Math.max(0, 1 - particle.age / particle.life);
    particle.x += particle.vx;
    particle.y += particle.vy;
    particle.vx *= 0.92;
    particle.vy *= 0.92;
    particle.age += 1;

    ctx.shadowBlur = 14;
    ctx.shadowColor = `hsla(${particle.hue}, 100%, 62%, ${opacity})`;
    ctx.fillStyle = `hsla(${particle.hue}, 100%, 62%, ${opacity})`;
    ctx.beginPath();
    ctx.arc(particle.x, particle.y, particle.radius * opacity, 0, Math.PI * 2);
    ctx.fill();
  }

  illusionTrails = illusionTrails.filter((trail) => trail.age < trail.life);
  illusionParticles = illusionParticles.filter(
    (particle) => particle.age < particle.life && particle.x >= -20 && particle.x <= width + 20 && particle.y >= -20 && particle.y <= height + 20
  );

  ctx.restore();
}

function triggerAction(prediction) {
  if (cooldown > 0) {
    cooldown -= 1;
    return actions[prediction.name] || actions.unknown;
  }

  cooldown = 18;

  if (prediction.name === "swipe_left") {
    screenIndex = (screenIndex - 1 + screens.length) % screens.length;
    activeFilter = (activeFilter - 1 + filters.length) % filters.length;
    canvas.style.filter = filters[activeFilter];
  }

  if (prediction.name === "swipe_right") {
    screenIndex = (screenIndex + 1) % screens.length;
    activeFilter = (activeFilter + 1) % filters.length;
    canvas.style.filter = filters[activeFilter];
  }

  if (prediction.name === "peace") {
    downloadScreenshot();
  }

  return actions[prediction.name] || actions.unknown;
}

function drawEmptyState() {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#050711");
  gradient.addColorStop(0.58, "#0f172a");
  gradient.addColorStop(1, "#111827");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(110, 231, 255, 0.1)";
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 42) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += 42) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  const cardWidth = Math.min(520, width - 48);
  const cardHeight = 168;
  const x = (width - cardWidth) / 2;
  const y = (height - cardHeight) / 2;

  ctx.fillStyle = "rgba(8, 13, 24, 0.88)";
  roundRect(ctx, x, y, cardWidth, cardHeight, 10);
  ctx.fill();
  ctx.strokeStyle = "rgba(110, 231, 255, 0.28)";
  ctx.stroke();

  ctx.fillStyle = "#36f06f";
  ctx.font = "800 12px Consolas, monospace";
  ctx.fillText("WEBCAM PREVIEW", x + 28, y + 38);

  ctx.fillStyle = "#f8fbff";
  ctx.font = "850 28px Inter, system-ui";
  ctx.fillText("Start camera to detect gestures", x + 28, y + 78);

  ctx.fillStyle = "#8ea0b8";
  ctx.font = "650 16px Inter, system-ui";
  ctx.fillText("Allow camera access, then show your hand in the frame.", x + 28, y + 112);
  ctx.fillText("Landmarks and predictions will appear here live.", x + 28, y + 138);
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.arcTo(x + width, y, x + width, y + height, radius);
  context.arcTo(x + width, y + height, x, y + height, radius);
  context.arcTo(x, y + height, x, y, radius);
  context.arcTo(x, y, x + width, y, radius);
  context.closePath();
}

function onResults(results) {
  const width = canvas.width;
  const height = canvas.height;
  ctx.save();
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(results.image, 0, 0, width, height);

  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    motionHistory = [];
    lastIllusionPoint = null;
    drawMotionIllusions(width, height);
    updateUi("No hand detected", 0, "Waiting for hand");
    ctx.restore();
    return;
  }

  for (const landmarks of results.multiHandLandmarks) {
    addMotionIllusions(landmarks, width, height);
  }
  drawMotionIllusions(width, height);

  for (const landmarks of results.multiHandLandmarks) {
    window.drawConnectors(ctx, landmarks, window.HAND_CONNECTIONS, {
      color: "#25e85f",
      lineWidth: 3
    });
    window.drawLandmarks(ctx, landmarks, {
      color: "#ff2f55",
      lineWidth: 2,
      radius: 4
    });
  }

  const landmarks = results.multiHandLandmarks[0];
  const staticPrediction = classifyStatic(landmarks);
  const motionPrediction = classifyMotion(landmarks);
  const prediction = motionPrediction && motionPrediction.confidence >= 0.5 ? motionPrediction : staticPrediction;
  const action = prediction.confidence >= 0.55 ? triggerAction(prediction) : "Show response";

  if (isRecording) {
    samples.push(landmarkRow(gestureLabel.value, landmarks));
    sampleCount.textContent = `${samples.length} samples`;
  }

  updateUi(prediction.name, prediction.confidence, action);
  ctx.restore();
}

async function startCamera() {
  if (!window.Hands || !window.Camera) {
    setStatus("MediaPipe failed to load");
    actionName.textContent = "Check your internet connection";
    return;
  }

  resizeCanvas();
  drawEmptyState();

  hands = new window.Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.65,
    minTrackingConfidence: 0.6
  });

  hands.onResults(onResults);

  camera = new window.Camera(video, {
    width: 1280,
    height: 720,
    onFrame: async () => {
      await hands.send({ image: video });
    }
  });

  try {
    await camera.start();
    isRunning = true;
    setStatus("Camera live");
    startButton.disabled = true;
    updateUi("Ready", 0, "Show your hand");
  } catch (error) {
    setStatus("Camera blocked");
    updateUi("Camera blocked", 0, "Allow webcam access and reload");
  }
}

function stopCamera() {
  if (camera) {
    camera.stop();
  }
  isRunning = false;
  isRecording = false;
  motionHistory = [];
  illusionTrails = [];
  illusionParticles = [];
  lastIllusionPoint = null;
  startButton.disabled = false;
  recordPanel.classList.remove("is-recording");
  recordingState.textContent = "Not recording";
  setStatus("Camera off");
  updateUi("Stopped", 0, "Camera stopped");
}

function toggleRecording() {
  if (!isRunning) {
    actionName.textContent = "Start the camera first";
    return;
  }
  isRecording = !isRecording;
  recordPanel.classList.toggle("is-recording", isRecording);
  recordingState.textContent = isRecording ? "Recording..." : "Not recording";
}

function downloadCsv() {
  if (samples.length === 0) {
    actionName.textContent = "No samples recorded yet";
    return;
  }

  const header = ["label"];
  for (let index = 0; index < 21; index += 1) {
    header.push(`x${index}`, `y${index}`, `z${index}`);
  }

  const csv = [header, ...samples]
    .map((row) => row.join(","))
    .join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `gestureflow_${gestureLabel.value}_samples.csv`;
  link.click();
  URL.revokeObjectURL(url);
  isRecording = false;
  recordPanel.classList.remove("is-recording");
  recordingState.textContent = "Not recording";
}

function downloadScreenshot() {
  const link = document.createElement("a");
  link.download = "gestureflow_screenshot.png";
  link.href = canvas.toDataURL("image/png");
  link.click();
}

startButton.addEventListener("click", startCamera);
recordButton.addEventListener("click", toggleRecording);
saveButton.addEventListener("click", downloadCsv);
stopButton.addEventListener("click", stopCamera);

window.addEventListener("resize", () => {
  resizeCanvas();
  if (!isRunning) {
    drawEmptyState();
  }
});

window.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  if (key === "r") {
    toggleRecording();
  }
  if (key === "s") {
    downloadCsv();
  }
  if (key === "q") {
    stopCamera();
  }
});

resizeCanvas();
drawEmptyState();
updateUi(lastGesture, lastConfidence, lastAction);
