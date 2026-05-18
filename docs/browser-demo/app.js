"use strict";

const ORT_CDN_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
const DEFAULT_MODEL_URL = "./assets/inherent.onnx";
const DEFAULT_METADATA_URL = "./assets/inherent.onnx.metadata.json";
const FEEDBACK_STORAGE_KEY = "inherent.browserDemo.feedbackExamples";
const SAMPLE_RATE = 16000;
const MAX_SECONDS = 60;
const MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS;
const DEFAULT_INFERENCE_WINDOW_SECONDS = 2.5;
const DEFAULT_MAX_FRAMES = 3000;
const MEL_BINS = 128;
const HOP_SAMPLES = 320;
const WINDOW_SAMPLES = 400;
const FFT_SIZE = 512;
const EPSILON = 1e-8;

const FALLBACK_HEAD_ORDER = [
  "isInteresting",
  "hasAddToListIntent",
  "hasTermSearchQuery",
  "hasPhotoQuery",
  "hasCalendarEvent",
  "hasCreateDocIntent",
  "hasPersonContext",
  "hasEventContext",
  "hasDeepResearchIntent",
  "hasInsightIntent",
  "hasBrowsingAgentIntent",
  "hasCallingAgentIntent",
  "hasStartTimerIntent",
];

const FALLBACK_THRESHOLD_KEYS = [
  "is_interesting",
  "has_add_to_list_intent",
  "has_term_search_query",
  "has_photo_query",
  "has_calendar_event",
  "has_create_doc_intent",
  "has_person_context",
  "has_event_context",
  "has_deep_research_intent",
  "has_insight_intent",
  "has_browsing_agent_intent",
  "has_calling_agent_intent",
  "has_start_timer_intent",
];

const FALLBACK_THRESHOLDS = {
  is_interesting: 0.57,
  has_add_to_list_intent: 0.65,
  has_term_search_query: 0.90,
  has_photo_query: 0.90,
  has_calendar_event: 0.90,
  has_create_doc_intent: 0.90,
  has_person_context: 0.56,
  has_event_context: 0.74,
  has_deep_research_intent: 0.90,
  has_insight_intent: 0.90,
  has_browsing_agent_intent: 0.90,
  has_calling_agent_intent: 0.90,
  has_start_timer_intent: 0.90,
};

const state = {
  session: null,
  metadata: null,
  provider: null,
  audioContext: null,
  mediaStream: null,
  sourceNode: null,
  processorNode: null,
  chunks: [],
  inputSampleRate: SAMPLE_RATE,
  recording: false,
  inferenceTimer: null,
  inferInFlight: false,
  melFilterBank: null,
  hannWindow: null,
  lastScores: [],
  lastRecordingStats: null,
  pendingFeedbackHead: null,
  feedbackExamples: loadFeedbackExamples(),
};

const dom = {
  modelUrl: document.getElementById("modelUrl"),
  metadataUrl: document.getElementById("metadataUrl"),
  modelFile: document.getElementById("modelFile"),
  metadataFile: document.getElementById("metadataFile"),
  loadModel: document.getElementById("loadModel"),
  resetDemo: document.getElementById("resetDemo"),
  startRecording: document.getElementById("startRecording"),
  stopRecording: document.getElementById("stopRecording"),
  modelStatus: document.getElementById("modelStatus"),
  modelLoader: document.getElementById("modelLoader"),
  modelProgress: document.querySelector(".model-progress"),
  modelProgressFill: document.getElementById("modelProgressFill"),
  modelProgressText: document.getElementById("modelProgressText"),
  recordingStatus: document.getElementById("recordingStatus"),
  runtimeBadge: document.getElementById("runtimeBadge"),
  recordingBadge: document.getElementById("recordingBadge"),
  inferenceBadge: document.getElementById("inferenceBadge"),
  levelFill: document.getElementById("levelFill"),
  scope: document.getElementById("scope"),
  bestMatch: document.getElementById("bestMatch"),
  heads: document.getElementById("heads"),
  flowHeads: document.getElementById("flowHeads"),
  gateScore: document.getElementById("gateScore"),
  qualityGates: document.getElementById("qualityGates"),
  feedbackTranscript: document.getElementById("feedbackTranscript"),
  feedbackConsent: document.getElementById("feedbackConsent"),
  feedbackStatus: document.getElementById("feedbackStatus"),
  feedbackCount: document.getElementById("feedbackCount"),
  downloadFeedback: document.getElementById("downloadFeedback"),
  clearFeedback: document.getElementById("clearFeedback"),
  headTemplate: document.getElementById("headTemplate"),
  flowHeadTemplate: document.getElementById("flowHeadTemplate"),
};

if (globalThis.ort) {
  ort.env.wasm.wasmPaths = ORT_CDN_BASE;
}

renderHeadCards();
renderFlowHeads();
renderFeedbackCount();
updateQualityGates();
drawIdleScope();
wireEvents();
autoLoadDefaultModel();

function wireEvents() {
  if (dom.loadModel) {
    dom.loadModel.addEventListener("click", () => loadModel());
  }
  if (dom.resetDemo) {
    dom.resetDemo.addEventListener("click", () => resetDemo());
  }
  dom.startRecording.addEventListener("click", () => startRecording());
  dom.stopRecording.addEventListener("click", () => stopRecording(true));
  if (dom.downloadFeedback) {
    dom.downloadFeedback.addEventListener("click", () => downloadFeedbackExamples());
  }
  if (dom.clearFeedback) {
    dom.clearFeedback.addEventListener("click", () => clearFeedbackExamples());
  }
  for (const input of document.querySelectorAll('input[name="feedbackKind"]')) {
    input.addEventListener("change", () => updateQualityGates());
  }
  if (dom.feedbackTranscript) {
    dom.feedbackTranscript.addEventListener("input", () => updateQualityGates());
  }
  if (dom.feedbackConsent) {
    dom.feedbackConsent.addEventListener("change", () => updateQualityGates());
  }
}

async function autoLoadDefaultModel() {
  try {
    await loadModel({ quiet404: true });
  } catch (error) {
    updateRuntimeBadge("Model: unavailable", false);
    updateModelProgress(100, "Failed", "failed");
    setModelStatus(`Bundled model failed to load: ${error.message}`);
    dom.startRecording.disabled = true;
    setRecordingStatus("Recording is disabled because the bundled model could not be loaded.");
  }
}

async function loadModel(options = {}) {
  ensureOrtLoaded();
  updateModelProgress(2, "Starting");
  setModelStatus("Loading bundled model...");
  dom.startRecording.disabled = true;
  if (dom.loadModel) {
    dom.loadModel.disabled = true;
  }

  try {
    updateModelProgress(8, "Metadata");
    state.metadata = await loadMetadata(options);
    updateModelProgress(15, "Downloading model");
    const modelSource = await loadModelSource(options);
    updateModelProgress(86, "Starting runtime");
    state.session = await createSessionWithFallback(modelSource);
    updateModelProgress(100, "Ready", "ready");
    renderHeadCards();
    renderFlowHeads();
    updateQualityGates();
    updateRuntimeBadge(`Model: ${state.provider}`, true);
    dom.startRecording.disabled = false;
    setModelStatus(
      `Bundled ${state.metadata.artifact_format || "onnx"} model ready.`,
    );
    setRecordingStatus("Ready. Press record and speak into the browser mic.");
  } finally {
    if (dom.loadModel) {
      dom.loadModel.disabled = false;
    }
  }
}

async function loadModelSource(options) {
  const file = dom.modelFile && dom.modelFile.files && dom.modelFile.files[0];
  if (file) {
    updateModelProgress(30, "Reading local model");
    const buffer = await file.arrayBuffer();
    updateModelProgress(82, formatLoadedBytes(file.size, file.size));
    return buffer;
  }

  const url = dom.modelUrl && dom.modelUrl.value.trim()
    ? dom.modelUrl.value.trim()
    : DEFAULT_MODEL_URL;
  const response = await fetch(url);
  if (!response.ok) {
    const message = `model fetch failed (${response.status}) for ${url}`;
    if (options.quiet404 && response.status === 404) {
      throw new Error(message);
    }
    throw new Error(message);
  }
  return readResponseWithProgress(response);
}

async function readResponseWithProgress(response) {
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !response.body.getReader) {
    const buffer = await response.arrayBuffer();
    updateModelProgress(82, formatLoadedBytes(buffer.byteLength, total || buffer.byteLength));
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    received += value.byteLength;
    if (total > 0) {
      const downloadProgress = 15 + (received / total) * 67;
      updateModelProgress(downloadProgress, formatLoadedBytes(received, total));
    } else {
      const estimatedProgress = Math.min(78, 18 + chunks.length * 3);
      updateModelProgress(estimatedProgress, formatLoadedBytes(received, 0));
    }
  }

  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  updateModelProgress(82, formatLoadedBytes(received, total || received));
  return bytes.buffer;
}

async function loadMetadata(options) {
  const file = dom.metadataFile && dom.metadataFile.files && dom.metadataFile.files[0];
  if (file) {
    return JSON.parse(await file.text());
  }

  const url = dom.metadataUrl && dom.metadataUrl.value.trim()
    ? dom.metadataUrl.value.trim()
    : DEFAULT_METADATA_URL;
  const response = await fetch(url);
  if (response.status === 404 && options.quiet404) {
    return fallbackMetadata();
  }
  if (!response.ok) {
    throw new Error(`metadata fetch failed (${response.status}) for ${url}`);
  }
  return response.json();
}

async function createSessionWithFallback(modelSource) {
  const providers = browserExecutionProviders();
  let lastError = null;

  for (const provider of providers) {
    try {
      const session = await ort.InferenceSession.create(modelSource, {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      });
      state.provider = provider;
      return session;
    } catch (error) {
      lastError = error;
      console.warn(`ONNX Runtime Web provider ${provider} failed`, error);
    }
  }

  throw new Error(`unable to create ONNX Runtime Web session: ${lastError.message}`);
}

function browserExecutionProviders() {
  const providers = [];
  if ("gpu" in navigator) {
    providers.push("webgpu");
  }
  providers.push("wasm");
  return providers;
}

async function startRecording() {
  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
      },
    });
    state.audioContext = new AudioContext();
    state.inputSampleRate = state.audioContext.sampleRate;
    state.sourceNode = state.audioContext.createMediaStreamSource(state.mediaStream);
    state.processorNode = state.audioContext.createScriptProcessor(4096, 1, 1);
    state.chunks = [];
    state.lastRecordingStats = null;
    state.pendingFeedbackHead = null;
    state.recording = true;
    updateQualityGates();

    state.processorNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const chunk = new Float32Array(input);
      state.chunks.push(chunk);
      trimBufferedChunks();
      updateLevel(chunk);
      drawScope(chunk);
    };

    state.sourceNode.connect(state.processorNode);
    state.processorNode.connect(state.audioContext.destination);

    dom.startRecording.disabled = true;
    dom.stopRecording.disabled = false;
    updateRecordingBadge("Mic: recording", true);
    setRecordingStatus(state.session
      ? "Recording... scores update about once per second."
      : "Recording mic preview. Load an ONNX model to enable head-hit inference.");

    state.inferenceTimer = window.setInterval(() => {
      runInferenceFromBufferedAudio().catch((error) => {
        setInferenceStatus(`Inference error: ${error.message}`);
      });
    }, 1000);
  } catch (error) {
    setRecordingStatus(`Microphone error: ${error.message}`);
    await stopRecording(false);
  }
}

async function stopRecording(runFinalInference) {
  window.clearInterval(state.inferenceTimer);
  state.inferenceTimer = null;
  state.recording = false;

  if (state.processorNode) {
    state.processorNode.disconnect();
    state.processorNode.onaudioprocess = null;
  }
  if (state.sourceNode) {
    state.sourceNode.disconnect();
  }
  if (state.audioContext) {
    await state.audioContext.close();
  }
  if (state.mediaStream) {
    for (const track of state.mediaStream.getTracks()) {
      track.stop();
    }
  }

  state.processorNode = null;
  state.sourceNode = null;
  state.audioContext = null;
  state.mediaStream = null;
  dom.startRecording.disabled = false;
  dom.stopRecording.disabled = true;
  updateRecordingBadge("Mic: idle", false);
  dom.levelFill.style.width = "0%";
  drawIdleScope();

  if (runFinalInference && state.chunks.length) {
    state.lastRecordingStats = analyzeAudio(mergeChunks(state.chunks), state.inputSampleRate);
    await runInferenceFromBufferedAudio();
    setRecordingStatus("Recording stopped. Final scores shown below.");
    updateQualityGates();
  } else {
    setRecordingStatus(state.session
      ? "Ready to record from the browser microphone."
      : "Start recording to preview the mic oscilloscope. Load a model to see head hits.");
    updateQualityGates();
  }
}

async function runInferenceFromBufferedAudio() {
  if (!state.session || state.inferInFlight || !state.chunks.length) {
    return;
  }

  state.inferInFlight = true;
  setInferenceStatus("Inference: running");
  try {
    const inputAudio = recentAudioWindow(mergeChunks(state.chunks));
    if (inputAudio.length < Math.floor(state.inputSampleRate * 0.2)) {
      return;
    }
    const audio16k = resampleLinear(inputAudio, state.inputSampleRate, SAMPLE_RATE);
    const mel = audioToMel(audio16k);
    const tensor = melToTensor(mel);
    const feeds = { [inputName()]: tensor };
    const outputs = await state.session.run(feeds);
    const output = outputs[outputName()] || outputs[state.session.outputNames[0]];
    renderScores(Array.from(output.data));
    setInferenceStatus("Inference: updated");
  } finally {
    state.inferInFlight = false;
  }
}

function audioToMel(audio) {
  const limited = audio.length > MAX_SAMPLES ? audio.slice(audio.length - MAX_SAMPLES) : audio;
  const frameCount = Math.min(
    maxFrames(),
    Math.max(1, Math.ceil(limited.length / HOP_SAMPLES)),
  );
  const mel = new Float32Array(frameCount * MEL_BINS);

  state.hannWindow = state.hannWindow || makeHannWindow(WINDOW_SAMPLES);
  state.melFilterBank = state.melFilterBank || makeMelFilterBank();

  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * HOP_SAMPLES;
    const spectrum = powerSpectrum(limited, offset, state.hannWindow);
    for (let bin = 0; bin < MEL_BINS; bin += 1) {
      const filter = state.melFilterBank[bin];
      let energy = 0;
      for (let k = filter.start; k < filter.end; k += 1) {
        energy += spectrum[k] * filter.weights[k - filter.start];
      }
      mel[frame * MEL_BINS + bin] = Math.log(energy + EPSILON);
    }
  }

  return { data: mel, frames: frameCount };
}

function melToTensor(mel) {
  const targetFrames = tensorFrameCount(mel.frames);
  const data = new Float32Array(targetFrames * MEL_BINS);
  data.fill(browserPaddingValue());
  const framesToCopy = Math.min(targetFrames, mel.frames);
  data.set(mel.data.slice(0, framesToCopy * MEL_BINS));
  return new ort.Tensor("float32", data, [1, targetFrames, MEL_BINS]);
}

function browserPaddingValue() {
  const configured = state.metadata &&
    state.metadata.config &&
    state.metadata.config.export &&
    state.metadata.config.export.browser_padding_value;
  return typeof configured === "number" ? configured : 0;
}

function tensorFrameCount(actualFrames) {
  const staticFrames = state.metadata &&
    state.metadata.config &&
    state.metadata.config.export &&
    state.metadata.config.export.onnx_static_frames;
  if (Number.isInteger(staticFrames) && staticFrames > 0) {
    return staticFrames;
  }
  return Math.max(1, Math.min(maxFrames(), actualFrames));
}

function maxFrames() {
  const configured = state.metadata &&
    state.metadata.config &&
    state.metadata.config.model &&
    state.metadata.config.model.max_frames;
  return Number.isInteger(configured) && configured > 0 ? configured : DEFAULT_MAX_FRAMES;
}

function powerSpectrum(audio, offset, window) {
  const real = new Float32Array(FFT_SIZE);
  const imag = new Float32Array(FFT_SIZE);
  for (let i = 0; i < WINDOW_SAMPLES; i += 1) {
    real[i] = (audio[offset + i] || 0) * window[i];
  }
  fft(real, imag);

  const spectrum = new Float32Array(FFT_SIZE / 2 + 1);
  for (let k = 0; k < spectrum.length; k += 1) {
    spectrum[k] = (real[k] * real[k] + imag[k] * imag[k]) / FFT_SIZE;
  }
  return spectrum;
}

function fft(real, imag) {
  const n = real.length;
  let j = 0;
  for (let i = 1; i < n; i += 1) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const tempReal = real[i];
      const tempImag = imag[i];
      real[i] = real[j];
      imag[i] = imag[j];
      real[j] = tempReal;
      imag[j] = tempImag;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const wLenReal = Math.cos(angle);
    const wLenImag = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wReal = 1;
      let wImag = 0;
      for (let k = 0; k < len / 2; k += 1) {
        const evenReal = real[i + k];
        const evenImag = imag[i + k];
        const oddReal = real[i + k + len / 2] * wReal - imag[i + k + len / 2] * wImag;
        const oddImag = real[i + k + len / 2] * wImag + imag[i + k + len / 2] * wReal;
        real[i + k] = evenReal + oddReal;
        imag[i + k] = evenImag + oddImag;
        real[i + k + len / 2] = evenReal - oddReal;
        imag[i + k + len / 2] = evenImag - oddImag;

        const nextReal = wReal * wLenReal - wImag * wLenImag;
        wImag = wReal * wLenImag + wImag * wLenReal;
        wReal = nextReal;
      }
    }
  }
}

function makeHannWindow(length) {
  const window = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (length - 1));
  }
  return window;
}

function makeMelFilterBank() {
  const minMel = hzToMel(20);
  const maxMel = hzToMel(SAMPLE_RATE / 2);
  const melPoints = new Array(MEL_BINS + 2);
  const fftBins = new Array(MEL_BINS + 2);

  for (let i = 0; i < melPoints.length; i += 1) {
    const mel = minMel + (i / (MEL_BINS + 1)) * (maxMel - minMel);
    melPoints[i] = melToHz(mel);
    fftBins[i] = Math.min(
      FFT_SIZE / 2,
      Math.floor(((FFT_SIZE + 1) * melPoints[i]) / SAMPLE_RATE),
    );
  }

  const filters = [];
  for (let m = 1; m <= MEL_BINS; m += 1) {
    const left = fftBins[m - 1];
    const center = Math.max(fftBins[m], left + 1);
    const right = Math.max(fftBins[m + 1], center + 1);
    const weights = [];
    for (let k = left; k < right; k += 1) {
      const weight = k < center
        ? (k - left) / (center - left)
        : (right - k) / (right - center);
      weights.push(Math.max(0, weight));
    }
    filters.push({ start: left, end: right, weights });
  }
  return filters;
}

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function resampleLinear(input, inputRate, outputRate) {
  if (inputRate === outputRate) {
    return input;
  }
  const outputLength = Math.max(1, Math.round(input.length * outputRate / inputRate));
  const output = new Float32Array(outputLength);
  const ratio = input.length / outputLength;

  for (let i = 0; i < outputLength; i += 1) {
    const position = i * ratio;
    const index = Math.floor(position);
    const nextIndex = Math.min(input.length - 1, index + 1);
    const mix = position - index;
    output[i] = input[index] * (1 - mix) + input[nextIndex] * mix;
  }
  return output;
}

function mergeChunks(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function recentAudioWindow(audio) {
  const maxSamples = Math.max(1, Math.round(browserInferenceWindowSeconds() * state.inputSampleRate));
  if (audio.length <= maxSamples) {
    return audio;
  }
  return audio.slice(audio.length - maxSamples);
}

function browserInferenceWindowSeconds() {
  const configured = state.metadata &&
    state.metadata.config &&
    state.metadata.config.export &&
    state.metadata.config.export.browser_inference_window_seconds;
  return typeof configured === "number" && configured > 0
    ? configured
    : DEFAULT_INFERENCE_WINDOW_SECONDS;
}

function trimBufferedChunks() {
  const maxInputSamples = Math.ceil(MAX_SECONDS * state.inputSampleRate);
  let total = state.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  while (total > maxInputSamples && state.chunks.length > 1) {
    total -= state.chunks.shift().length;
  }
}

function updateLevel(chunk) {
  let sum = 0;
  for (let i = 0; i < chunk.length; i += 1) {
    sum += chunk[i] * chunk[i];
  }
  const rms = Math.sqrt(sum / chunk.length);
  const percent = Math.min(100, Math.round(rms * 500));
  dom.levelFill.style.width = `${percent}%`;
}

function drawScope(chunk) {
  const canvas = dom.scope;
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  drawScopeGrid(ctx, width, height);
  ctx.lineWidth = 5;
  ctx.strokeStyle = "#60ff7f";
  ctx.shadowColor = "#60ff7f";
  ctx.shadowBlur = 14;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(chunk.length / width));
  for (let x = 0; x < width; x += 1) {
    const sample = chunk[Math.min(chunk.length - 1, x * step)] || 0;
    const y = height / 2 - sample * height * 0.42;
    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function drawIdleScope() {
  const canvas = dom.scope;
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  drawScopeGrid(ctx, width, height);
  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(255, 233, 15, 0.85)";
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const y = height / 2 + Math.sin(x / 28) * 18;
    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}

function drawScopeGrid(ctx, width, height) {
  ctx.fillStyle = "#10151f";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(96, 255, 127, 0.16)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= width; x += 48) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y += 44) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255, 233, 15, 0.45)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();
}

function renderHeadCards() {
  const heads = headOrder();
  dom.heads.replaceChildren();
  for (const head of heads) {
    const node = dom.headTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.head = head;
    node.querySelector("strong").textContent = prettyHead(head);
    node.querySelector("span").textContent = "0.00";
    dom.heads.appendChild(node);
  }
}

function renderFlowHeads() {
  if (!dom.flowHeads) {
    return;
  }
  const intents = headOrder().slice(1);
  dom.flowHeads.replaceChildren();
  for (const head of intents) {
    const node = dom.flowHeadTemplate.content.firstElementChild.cloneNode(true);
    node.dataset.head = head;
    node.querySelector("strong").textContent = compactHead(head);
    node.querySelector("small").textContent = "0.00";
    node.querySelector(".donate-head-button").addEventListener("click", () => donateHeadLabel(head));
    dom.flowHeads.appendChild(node);
  }
  updateGateScore(null, null);
}

function renderScores(scores) {
  const heads = headOrder();
  const thresholds = thresholdValues();
  let best = null;
  const matches = [];
  state.lastScores = scores;

  heads.forEach((head, index) => {
    const score = Number(scores[index] || 0);
    const threshold = thresholds[index] || 0.5;
    const card = dom.heads.querySelector(`[data-head="${head}"]`);
    if (card) {
      card.classList.toggle("matched", score >= threshold);
      card.querySelector("span").textContent = `${score.toFixed(2)} / ${threshold.toFixed(2)}`;
      card.querySelector("i").style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
    }
    if (score >= threshold) {
      matches.push({ head, score, threshold, index });
    }
    if (!best || score > best.score) {
      best = { head, score, threshold, index };
    }
  });

  updateFlowScores(scores, thresholds);
  updateQualityGates();
  const intentMatches = matches.filter((match) => match.index > 0);
  const selected = intentMatches.sort((a, b) => b.score - a.score)[0] ||
    matches.sort((a, b) => b.score - a.score)[0] ||
    best;
  const matched = matches.includes(selected);
  updateBestMatch(selected, matched);
}

function updateFlowScores(scores, thresholds) {
  updateGateScore(Number(scores[0] || 0), thresholds[0] || 0.5);
  const heads = headOrder();
  for (let index = 1; index < heads.length; index += 1) {
    const head = heads[index];
    const score = Number(scores[index] || 0);
    const threshold = thresholds[index] || 0.5;
    const node = dom.flowHeads && dom.flowHeads.querySelector(`[data-head="${head}"]`);
    if (!node) {
      continue;
    }
    node.classList.toggle("hit", score >= threshold);
    node.querySelector("small").textContent = `${score.toFixed(2)} / ${threshold.toFixed(2)}`;
  }
}

function updateGateScore(score, threshold) {
  if (!dom.gateScore) {
    return;
  }
  if (score === null || threshold === null) {
    dom.gateScore.textContent = "waiting";
    return;
  }
  dom.gateScore.textContent = `${score.toFixed(2)} / ${threshold.toFixed(2)}`;
}

function updateBestMatch(match, matched) {
  const label = dom.bestMatch.querySelector(".label");
  const name = dom.bestMatch.querySelector("strong");
  const score = dom.bestMatch.querySelector(".score");
  label.textContent = matched ? "Matched above threshold" : "Highest score, below threshold";
  name.textContent = prettyHead(match.head);
  score.textContent = `${match.score.toFixed(3)} score, ${match.threshold.toFixed(3)} threshold`;
}

function updateModelProgress(percent, text, stateClass) {
  const clamped = Math.max(0, Math.min(100, Math.round(percent)));
  if (dom.modelProgressFill) {
    dom.modelProgressFill.style.width = `${clamped}%`;
  }
  if (dom.modelProgress) {
    dom.modelProgress.setAttribute("aria-valuenow", String(clamped));
  }
  if (dom.modelProgressText && text) {
    dom.modelProgressText.textContent = `${clamped}% - ${text}`;
  }
  if (dom.modelLoader) {
    dom.modelLoader.classList.toggle("ready", stateClass === "ready");
    dom.modelLoader.classList.toggle("failed", stateClass === "failed");
  }
}

function formatLoadedBytes(received, total) {
  if (total > 0) {
    return `${formatBytes(received)} / ${formatBytes(total)}`;
  }
  return `${formatBytes(received)} loaded`;
}

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function analyzeAudio(audio, sampleRate) {
  let sumSquares = 0;
  let peak = 0;
  for (let i = 0; i < audio.length; i += 1) {
    const sample = Math.abs(audio[i]);
    sumSquares += sample * sample;
    peak = Math.max(peak, sample);
  }
  return {
    duration_s: audio.length / sampleRate,
    peak,
    rms: audio.length ? Math.sqrt(sumSquares / audio.length) : 0,
    sample_rate: sampleRate,
    samples: audio.length,
  };
}

function updateQualityGates() {
  if (!dom.qualityGates) {
    return;
  }
  const validation = validateFeedbackExample();
  dom.qualityGates.replaceChildren();
  for (const gate of validation.gates) {
    const item = document.createElement("li");
    item.className = gate.pass ? "pass" : "fail";
    item.textContent = `${gate.pass ? "OK" : "Needs"} - ${gate.text}`;
    dom.qualityGates.append(item);
  }
  if (dom.feedbackStatus && validation.summary) {
    dom.feedbackStatus.textContent = validation.summary;
  }
}

function donateHeadLabel(head) {
  state.pendingFeedbackHead = head;
  const validation = validateFeedbackExample();
  updateQualityGates();
  if (!validation.passed) {
    dom.feedbackStatus.textContent =
      `Cannot save ${compactHead(head)} yet. Complete the validation gates first.`;
    return;
  }
  saveFeedbackExample();
}

function validateFeedbackExample() {
  const stats = state.lastRecordingStats;
  const kind = feedbackKind();
  const selectedHeads = selectedFeedbackHeads();
  const transcript = feedbackTranscript();
  const gates = [
    {
      pass: Boolean(state.session),
      text: "bundled model is loaded",
    },
    {
      pass: Boolean(stats) && !state.recording,
      text: "a recording has finished",
    },
    {
      pass: Boolean(stats) && stats.duration_s >= 0.5 && stats.duration_s <= 15,
      text: "clip duration is between 0.5s and 15s",
    },
    {
      pass: Boolean(stats) && stats.rms >= 0.003,
      text: "audio is not silent",
    },
    {
      pass: Boolean(stats) && stats.peak <= 0.99,
      text: "audio is not clipped",
    },
    {
      pass: state.lastScores.length === headOrder().length,
      text: "model scores are available for the recording",
    },
    {
      pass: transcript.length >= 3,
      text: "short transcript or note is provided",
    },
    {
      pass: kind === "negative" || selectedHeads.length > 0,
      text: "a route head donate button was selected",
    },
    {
      pass: Boolean(dom.feedbackConsent && dom.feedbackConsent.checked),
      text: "recording consent is checked",
    },
  ];
  const passed = gates.every((gate) => gate.pass);
  return {
    passed,
    gates,
    summary: passed
      ? "Ready to save a lightly validated training example."
      : "Complete the validation gates before saving this example.",
  };
}

function saveFeedbackExample() {
  const validation = validateFeedbackExample();
  updateQualityGates();
  if (!validation.passed) {
    return;
  }

  const stats = state.lastRecordingStats;
  const kind = feedbackKind();
  const selectedHeads = selectedFeedbackHeads();
  const example = {
    schema_version: 1,
    created_at: new Date().toISOString(),
    label_kind: kind,
    positive_heads: kind === "positive" ? selectedHeads : [],
    negative_heads: kind === "negative" ? selectedHeads : [],
    transcript_or_note: feedbackTranscript(),
    audio_summary: {
      duration_s: Number(stats.duration_s.toFixed(3)),
      rms: Number(stats.rms.toFixed(6)),
      peak: Number(stats.peak.toFixed(6)),
      sample_rate: stats.sample_rate,
      samples: stats.samples,
    },
    model: {
      version: state.metadata && state.metadata.version,
      training_hash: state.metadata && state.metadata.training_hash,
      provider: state.provider,
    },
    scores: scoreMap(),
    thresholds: thresholdMap(),
  };
  example.example_id = feedbackFingerprint(example);

  if (state.feedbackExamples.some((item) => item.example_id === example.example_id)) {
    dom.feedbackStatus.textContent = "Skipped duplicate example already saved in this browser.";
    return;
  }

  state.feedbackExamples.push(example);
  persistFeedbackExamples();
  renderFeedbackCount();
  dom.feedbackStatus.textContent = "Saved validated example locally. Download JSONL when ready to review.";
}

function downloadFeedbackExamples() {
  if (!state.feedbackExamples.length) {
    dom.feedbackStatus.textContent = "No saved examples yet.";
    return;
  }
  const jsonl = state.feedbackExamples.map((example) => JSON.stringify(example)).join("\n") + "\n";
  const blob = new Blob([jsonl], { type: "application/x-ndjson" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `inherent-feedback-${new Date().toISOString().slice(0, 10)}.jsonl`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  dom.feedbackStatus.textContent = `Downloaded ${state.feedbackExamples.length} saved example(s).`;
}

function clearFeedbackExamples() {
  state.feedbackExamples = [];
  persistFeedbackExamples();
  renderFeedbackCount();
  if (dom.feedbackStatus) {
    dom.feedbackStatus.textContent = "Cleared saved examples from this browser.";
  }
}

function feedbackKind() {
  const checked = document.querySelector('input[name="feedbackKind"]:checked');
  return checked ? checked.value : "positive";
}

function selectedFeedbackHeads() {
  return state.pendingFeedbackHead ? [state.pendingFeedbackHead] : [];
}

function feedbackTranscript() {
  return dom.feedbackTranscript ? dom.feedbackTranscript.value.trim() : "";
}

function scoreMap() {
  const result = {};
  headOrder().forEach((head, index) => {
    result[head] = Number((Number(state.lastScores[index] || 0)).toFixed(6));
  });
  return result;
}

function thresholdMap() {
  const result = {};
  const thresholds = thresholdValues();
  headOrder().forEach((head, index) => {
    result[head] = Number((Number(thresholds[index] || 0)).toFixed(6));
  });
  return result;
}

function feedbackFingerprint(example) {
  return [
    example.label_kind,
    example.positive_heads.join(","),
    example.negative_heads.join(","),
    example.transcript_or_note.toLowerCase(),
    example.audio_summary.duration_s.toFixed(1),
    example.audio_summary.rms.toFixed(3),
    Object.values(example.scores).map((score) => score.toFixed(2)).join(","),
  ].join("|");
}

function loadFeedbackExamples() {
  try {
    const raw = localStorage.getItem(FEEDBACK_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function persistFeedbackExamples() {
  try {
    localStorage.setItem(FEEDBACK_STORAGE_KEY, JSON.stringify(state.feedbackExamples));
  } catch {
    if (dom.feedbackStatus) {
      dom.feedbackStatus.textContent = "Could not persist examples in local storage.";
    }
  }
}

function renderFeedbackCount() {
  if (dom.feedbackCount) {
    dom.feedbackCount.textContent = `${state.feedbackExamples.length} saved`;
  }
}

function headOrder() {
  return Array.isArray(state.metadata && state.metadata.head_order)
    ? state.metadata.head_order
    : FALLBACK_HEAD_ORDER;
}

function thresholdKeys() {
  return Array.isArray(state.metadata && state.metadata.threshold_keys_in_order)
    ? state.metadata.threshold_keys_in_order
    : FALLBACK_THRESHOLD_KEYS;
}

function thresholdValues() {
  const defaults = (state.metadata && state.metadata.default_thresholds) || FALLBACK_THRESHOLDS;
  return thresholdKeys().map((key) => Number(defaults[key] || 0.5));
}

function inputName() {
  return (state.metadata && state.metadata.input_tensor) ||
    (state.session && state.session.inputNames[0]) ||
    "mel_spectrogram";
}

function outputName() {
  return (state.metadata && state.metadata.output_tensor) ||
    (state.session && state.session.outputNames[0]) ||
    "intent_output";
}

function fallbackMetadata() {
  return {
    artifact_format: "onnx",
    input_tensor: "mel_spectrogram",
    output_tensor: "intent_output",
    head_order: FALLBACK_HEAD_ORDER,
    threshold_keys_in_order: FALLBACK_THRESHOLD_KEYS,
    default_thresholds: FALLBACK_THRESHOLDS,
    config: {
      model: {
        max_frames: DEFAULT_MAX_FRAMES,
        mel_bins: MEL_BINS,
      },
      export: {
        onnx_static_frames: DEFAULT_MAX_FRAMES,
      },
    },
  };
}

function prettyHead(head) {
  return head
    .replace(/^has/, "")
    .replace(/Intent$/, " intent")
    .replace(/Query$/, " query")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/^is Interesting$/, "is interesting");
}

function compactHead(head) {
  return prettyHead(head)
    .replace(" intent", "")
    .replace(" query", "")
    .replace("Agent", "agent");
}

function resetDemo() {
  stopRecording(false);
  state.session = null;
  state.metadata = null;
  state.provider = null;
  state.chunks = [];
  state.lastScores = [];
  state.lastRecordingStats = null;
  state.pendingFeedbackHead = null;
  renderHeadCards();
  renderFlowHeads();
  drawIdleScope();
  updateQualityGates();
  updateRuntimeBadge("Runtime: not loaded", false);
  updateRecordingBadge("Mic: idle", false);
  setInferenceStatus("Inference: idle");
  setModelStatus("Waiting for a model.");
  setRecordingStatus("Start recording to preview the mic oscilloscope. Load a model to see head hits.");
  dom.startRecording.disabled = false;
  dom.stopRecording.disabled = true;
}

function ensureOrtLoaded() {
  if (!globalThis.ort) {
    throw new Error("ONNX Runtime Web did not load from the CDN");
  }
}

function setModelStatus(message) {
  dom.modelStatus.textContent = message;
}

function setRecordingStatus(message) {
  dom.recordingStatus.textContent = message;
}

function setInferenceStatus(message) {
  dom.inferenceBadge.textContent = message;
}

function updateRuntimeBadge(message, ready) {
  dom.runtimeBadge.textContent = message;
  dom.runtimeBadge.classList.toggle("ready", ready);
}

function updateRecordingBadge(message, recording) {
  dom.recordingBadge.textContent = message;
  dom.recordingBadge.classList.toggle("recording", recording);
}
