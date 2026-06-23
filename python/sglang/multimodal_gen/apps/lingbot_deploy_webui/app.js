const $ = (id) => document.getElementById(id);

const CONTROL_KEYS = new Map([
  ["w", "w"],
  ["a", "a"],
  ["s", "s"],
  ["d", "d"],
  ["i", "i"],
  ["j", "j"],
  ["k", "k"],
  ["l", "l"],
  ["arrowup", "i"],
  ["arrowleft", "j"],
  ["arrowdown", "k"],
  ["arrowright", "l"],
]);

const MAX_FRAME_QUEUE = 96;

const els = {
  wsUrl: $("wsUrl"),
  startBtn: $("startBtn"),
  stopBtn: $("stopBtn"),
  statusText: $("statusText"),
  socketBadge: $("socketBadge"),
  runBadge: $("runBadge"),
  startForm: $("startForm"),
  imagePath: $("imagePath"),
  prompt: $("prompt"),
  width: $("width"),
  height: $("height"),
  fps: $("fps"),
  numFrames: $("numFrames"),
  seed: $("seed"),
  upscalingScale: $("upscalingScale"),
  enableUpscaling: $("enableUpscaling"),
  debugSaveVideo: $("debugSaveVideo"),
  autoStop: $("autoStop"),
  moveSpeed: $("moveSpeed"),
  rotateIK: $("rotateIK"),
  rotateJL: $("rotateJL"),
  debugVideoPath: $("debugVideoPath"),
  eventMode: $("eventMode"),
  eventChunk: $("eventChunk"),
  event1Text: $("event1Text"),
  event2Text: $("event2Text"),
  customEventId: $("customEventId"),
  triggerCustomEvent: $("triggerCustomEvent"),
  previewCanvas: $("previewCanvas"),
  emptyPreview: $("emptyPreview"),
  outputSize: $("outputSize"),
  frameCount: $("frameCount"),
  chunkCount: $("chunkCount"),
  latency: $("latency"),
  queueDepth: $("queueDepth"),
  droppedFrames: $("droppedFrames"),
  preSrPath: $("preSrPath"),
  postSrPath: $("postSrPath"),
  copyPreSr: $("copyPreSr"),
  copyPostSr: $("copyPostSr"),
  timingView: $("timingView"),
  log: $("log"),
  clearLog: $("clearLog"),
};

const ctx = els.previewCanvas.getContext("2d", { alpha: false });

const state = {
  socket: null,
  running: false,
  started: false,
  stopSent: false,
  outputWidth: 1664,
  outputHeight: 960,
  fps: 16,
  targetFrames: 117,
  bytesPerFrame: 1664 * 960 * 3,
  receivedFrames: 0,
  renderedFrames: 0,
  receivedChunks: 0,
  droppedFrames: 0,
  frameQueue: [],
  activeKeys: new Set(),
  renderTimer: 0,
  nextFrameAt: 0,
  rgbaBuffer: null,
};

function defaultWebSocketUrl() {
  if (window.location.protocol === "http:" || window.location.protocol === "https:") {
    const pageUrl = new URL(window.location.href);
    pageUrl.protocol = pageUrl.protocol === "https:" ? "wss:" : "ws:";
    const proxyMatch = pageUrl.pathname.match(/^(.*workflow_[^/:]+:)8000(\/.*)?$/);
    if (proxyMatch) {
      pageUrl.pathname = `${proxyMatch[1]}8001/v1/lingbot/realtime`;
    } else {
      pageUrl.pathname = "/v1/lingbot/realtime";
      if (pageUrl.port === "8000") pageUrl.port = "8001";
    }
    pageUrl.search = "";
    pageUrl.hash = "";
    return pageUrl.toString();
  }
  return "wss://127.0.0.1:8001/v1/lingbot/realtime";
}

function setBadge(el, text, kind = "") {
  el.textContent = text;
  el.classList.toggle("ok", kind === "ok");
  el.classList.toggle("warn", kind === "warn");
  el.classList.toggle("error", kind === "error");
}

function logLine(message) {
  const time = new Date().toLocaleTimeString();
  els.log.textContent += `[${time}] ${message}\n`;
  els.log.scrollTop = els.log.scrollHeight;
}

function positiveInt(input, fallback) {
  const value = Number.parseInt(input.value, 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function optionalFloat(input) {
  const value = Number.parseFloat(input.value);
  return Number.isFinite(value) ? value : null;
}

function setRunning(value) {
  state.running = value;
  els.startBtn.disabled = value;
  els.stopBtn.disabled = !value;
  setBadge(els.socketBadge, value ? "open" : "closed", value ? "ok" : "");
  setBadge(els.runBadge, value ? "running" : "idle", value ? "ok" : "");
  els.statusText.textContent = value ? "running" : "idle";
}

function resetCounters() {
  state.started = false;
  state.stopSent = false;
  state.receivedFrames = 0;
  state.renderedFrames = 0;
  state.receivedChunks = 0;
  state.droppedFrames = 0;
  state.frameQueue = [];
  state.activeKeys.clear();
  state.nextFrameAt = 0;
  window.clearTimeout(state.renderTimer);
  state.renderTimer = 0;
  els.frameCount.textContent = "0";
  els.chunkCount.textContent = "0";
  els.queueDepth.textContent = "0";
  els.droppedFrames.textContent = "0";
  els.latency.textContent = "-";
  els.timingView.textContent = "{}";
  els.preSrPath.value = "";
  els.postSrPath.value = "";
  updateControlButtons();
}

function collectEvents() {
  const events = {};
  const event1 = els.event1Text.value.trim();
  const event2 = els.event2Text.value.trim();
  if (event1) events.event1 = event1;
  if (event2) events.event2 = event2;
  return events;
}

function buildStartPayload() {
  const width = positiveInt(els.width, 1664);
  const height = positiveInt(els.height, 960);
  const fps = positiveInt(els.fps, 16);
  const numFrames = positiveInt(els.numFrames, 117);
  state.outputWidth = width;
  state.outputHeight = height;
  state.fps = fps;
  state.targetFrames = numFrames;
  state.bytesPerFrame = width * height * 3;
  const seedValue = Number.parseInt(els.seed.value, 10);

  const payload = {
    type: "START",
    stream_id: `lingbot_webui_${Date.now().toString(36)}`,
    prompt: els.prompt.value.trim(),
    image_path: els.imagePath.value.trim(),
    width,
    height,
    fps,
    num_frames: numFrames,
    seed: Number.isFinite(seedValue) ? seedValue : 42,
    enable_upscaling: els.enableUpscaling.checked,
    upscaling_scale: positiveInt(els.upscalingScale, 2),
    move_speed: optionalFloat(els.moveSpeed) ?? 0.05,
    rotate_speed_deg_ik: optionalFloat(els.rotateIK) ?? 4,
    rotate_speed_deg_jl: optionalFloat(els.rotateJL) ?? 6,
    event_mode: els.eventMode.value,
    event_chunk: positiveInt(els.eventChunk, 4),
  };

  const events = collectEvents();
  if (Object.keys(events).length > 0) payload.events = events;
  if (els.debugSaveVideo.checked) payload.debug_save_video = true;
  const debugVideoPath = els.debugVideoPath.value.trim();
  if (debugVideoPath) payload.debug_video_path = debugVideoPath;
  return payload;
}

function configureCanvas(width, height) {
  if (els.previewCanvas.width === width && els.previewCanvas.height === height) {
    return;
  }
  els.previewCanvas.width = width;
  els.previewCanvas.height = height;
  state.rgbaBuffer = new Uint8ClampedArray(width * height * 4);
  els.outputSize.textContent = `${width}x${height}`;
}

function sendJson(message) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return false;
  state.socket.send(JSON.stringify(message));
  return true;
}

function stopSession() {
  if (state.stopSent) return;
  state.stopSent = true;
  sendJson({ type: "STOP" });
  logLine("STOP");
}

function closeSocket() {
  if (state.socket && state.socket.readyState <= WebSocket.OPEN) {
    state.socket.close();
  }
}

function startSession() {
  const url = els.wsUrl.value.trim();
  if (!url) return;
  closeSocket();
  resetCounters();
  configureCanvas(state.outputWidth, state.outputHeight);

  const payload = buildStartPayload();
  const socket = new WebSocket(url);
  socket.binaryType = "arraybuffer";
  state.socket = socket;
  setRunning(true);
  setBadge(els.runBadge, "opening", "warn");
  els.statusText.textContent = "opening";

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify(payload));
    logLine(`START ${payload.stream_id}`);
    setBadge(els.runBadge, "starting", "warn");
  });

  socket.addEventListener("message", (event) => {
    if (typeof event.data === "string") {
      handleTextMessage(event.data);
    } else if (event.data instanceof ArrayBuffer) {
      handleFramePayload(event.data);
    } else if (event.data instanceof Blob) {
      event.data.arrayBuffer().then(handleFramePayload).catch((error) => {
        logLine(`blob decode error: ${error.message || error}`);
      });
    }
  });

  socket.addEventListener("error", () => {
    setBadge(els.runBadge, "error", "error");
    els.statusText.textContent = "websocket error";
    logLine("websocket error");
  });

  socket.addEventListener("close", () => {
    setRunning(false);
    releaseActiveKeys();
    logLine("socket closed");
  });
}

function handleTextMessage(text) {
  let message;
  try {
    message = JSON.parse(text);
  } catch {
    logLine(`text ${text}`);
    return;
  }

  const type = String(message.type || "").toUpperCase();
  if (type === "STARTED") {
    state.started = true;
    state.outputWidth = Number(message.output_width || message.width || state.outputWidth);
    state.outputHeight = Number(message.output_height || message.height || state.outputHeight);
    state.bytesPerFrame = state.outputWidth * state.outputHeight * 3;
    configureCanvas(state.outputWidth, state.outputHeight);
    els.preSrPath.value = message.debug_video_pre_sr_path || "";
    els.postSrPath.value = message.debug_video_post_sr_path || message.debug_video_path || "";
    setBadge(els.runBadge, "running", "ok");
    els.statusText.textContent = `seed ${message.seed ?? "-"}`;
    logLine(`STARTED output=${state.outputWidth}x${state.outputHeight}`);
    return;
  }

  if (type === "TIMING") {
    state.receivedChunks += 1;
    els.chunkCount.textContent = String(state.receivedChunks);
    els.latency.textContent = `${Number(message.chunk_time_ms || 0).toFixed(1)} ms`;
    els.timingView.textContent = JSON.stringify(message, null, 2);
    return;
  }

  if (type === "FINISH") {
    setBadge(els.runBadge, "complete", "ok");
    els.statusText.textContent = "complete";
    logLine("FINISH");
    closeSocket();
    return;
  }

  if (type === "ERROR") {
    setBadge(els.runBadge, "error", "error");
    els.statusText.textContent = message.message || "server error";
    logLine(`ERROR ${message.message || JSON.stringify(message)}`);
    return;
  }

  logLine(`${type || "message"} ${JSON.stringify(message)}`);
}

function handleFramePayload(buffer) {
  if (!state.bytesPerFrame) return;
  const frameCount = Math.floor(buffer.byteLength / state.bytesPerFrame);
  if (frameCount <= 0) {
    logLine(`binary payload ignored bytes=${buffer.byteLength}`);
    return;
  }
  if (buffer.byteLength % state.bytesPerFrame !== 0) {
    logLine(`binary payload has trailing bytes=${buffer.byteLength % state.bytesPerFrame}`);
  }

  for (let index = 0; index < frameCount; index += 1) {
    const offset = index * state.bytesPerFrame;
    state.frameQueue.push(new Uint8Array(buffer, offset, state.bytesPerFrame));
  }

  if (state.frameQueue.length > MAX_FRAME_QUEUE) {
    const dropCount = state.frameQueue.length - MAX_FRAME_QUEUE;
    state.frameQueue.splice(0, dropCount);
    state.droppedFrames += dropCount;
  }

  state.receivedFrames += frameCount;
  els.frameCount.textContent = String(state.receivedFrames);
  els.queueDepth.textContent = String(state.frameQueue.length);
  els.droppedFrames.textContent = String(state.droppedFrames);

  if (
    els.autoStop.checked &&
    !state.stopSent &&
    state.receivedFrames >= state.targetFrames
  ) {
    stopSession();
  }
  scheduleRender();
}

function scheduleRender() {
  if (state.renderTimer || state.frameQueue.length === 0) return;
  const now = performance.now();
  if (!state.nextFrameAt) state.nextFrameAt = now;
  const wait = Math.max(0, state.nextFrameAt - now);
  state.renderTimer = window.setTimeout(renderNextFrame, wait);
}

function renderNextFrame() {
  state.renderTimer = 0;
  const frame = state.frameQueue.shift();
  if (!frame) return;
  renderRgbFrame(frame);
  state.renderedFrames += 1;
  els.queueDepth.textContent = String(state.frameQueue.length);
  state.nextFrameAt = performance.now() + 1000 / Math.max(1, state.fps);
  if (state.frameQueue.length > 0) scheduleRender();
}

function renderRgbFrame(rgb) {
  const width = state.outputWidth;
  const height = state.outputHeight;
  const pixelCount = width * height;
  if (!state.rgbaBuffer || state.rgbaBuffer.length !== pixelCount * 4) {
    state.rgbaBuffer = new Uint8ClampedArray(pixelCount * 4);
  }
  const rgba = state.rgbaBuffer;
  for (let src = 0, dst = 0; src < rgb.length; src += 3, dst += 4) {
    rgba[dst] = rgb[src];
    rgba[dst + 1] = rgb[src + 1];
    rgba[dst + 2] = rgb[src + 2];
    rgba[dst + 3] = 255;
  }
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  els.emptyPreview.classList.add("hidden");
}

function isTypingTarget(target) {
  return ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function sendControl(key, action) {
  if (!state.running) return;
  if (!sendJson({ type: "CONTROL", key, action })) return;
  if (action === "down") state.activeKeys.add(key);
  else state.activeKeys.delete(key);
  updateControlButtons();
}

function releaseActiveKeys() {
  for (const key of Array.from(state.activeKeys)) {
    sendJson({ type: "CONTROL", key, action: "up" });
  }
  state.activeKeys.clear();
  updateControlButtons();
}

function updateControlButtons() {
  document.querySelectorAll("[data-key]").forEach((button) => {
    button.classList.toggle("active", state.activeKeys.has(button.dataset.key));
  });
}

function triggerEvent(eventId) {
  if (!eventId) return;
  if (sendJson({ type: "CONTROL", event: eventId })) {
    logLine(`CONTROL event=${eventId}`);
  }
}

function copyValue(input) {
  const value = input.value.trim();
  if (!value) return;
  navigator.clipboard.writeText(value).then(
    () => logLine(`copied ${value}`),
    () => logLine(`copy failed ${value}`),
  );
}

function bindControls() {
  els.startBtn.addEventListener("click", startSession);
  els.stopBtn.addEventListener("click", stopSession);
  els.clearLog.addEventListener("click", () => {
    els.log.textContent = "";
  });
  els.copyPreSr.addEventListener("click", () => copyValue(els.preSrPath));
  els.copyPostSr.addEventListener("click", () => copyValue(els.postSrPath));

  document.querySelectorAll("[data-event]").forEach((button) => {
    button.addEventListener("click", () => triggerEvent(button.dataset.event));
  });
  els.triggerCustomEvent.addEventListener("click", () => {
    triggerEvent(els.customEventId.value.trim());
  });

  document.querySelectorAll("[data-key]").forEach((button) => {
    const key = button.dataset.key;
    button.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      sendControl(key, "down");
      button.setPointerCapture(event.pointerId);
    });
    button.addEventListener("pointerup", (event) => {
      event.preventDefault();
      sendControl(key, "up");
    });
    button.addEventListener("pointercancel", () => sendControl(key, "up"));
    button.addEventListener("lostpointercapture", () => sendControl(key, "up"));
  });

  window.addEventListener("keydown", (event) => {
    if (event.repeat || isTypingTarget(event.target)) return;
    const key = CONTROL_KEYS.get(event.key.toLowerCase());
    if (!key || state.activeKeys.has(key)) return;
    event.preventDefault();
    sendControl(key, "down");
  });
  window.addEventListener("keyup", (event) => {
    if (isTypingTarget(event.target)) return;
    const key = CONTROL_KEYS.get(event.key.toLowerCase());
    if (!key) return;
    event.preventDefault();
    sendControl(key, "up");
  });

  window.addEventListener("blur", releaseActiveKeys);
  window.addEventListener("beforeunload", () => {
    releaseActiveKeys();
    if (state.running) stopSession();
  });
}

function init() {
  els.wsUrl.value = defaultWebSocketUrl();
  els.stopBtn.disabled = true;
  configureCanvas(state.outputWidth, state.outputHeight);
  setRunning(false);
  bindControls();
  logLine("ready");
}

init();
