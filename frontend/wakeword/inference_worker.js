/* global ort */
const ORT_SCRIPT_CANDIDATES = [
  '/vendor/ort/ort.wasm.min.js',
];

let ortScriptLoadError = null;
for (const url of ORT_SCRIPT_CANDIDATES) {
  try {
    self.importScripts(url);
    if (self.ort) {
      break;
    }
  } catch (error) {
    ortScriptLoadError = error;
  }
}

const FRAME_SIZE = 1280;

let melspecSession = null;
let embeddingSession = null;
let wakewordSessions = {};
let wakewordLabels = [];

let melBuffer = [];
let embeddingBuffer = [];
let thresholds = {};
let cooldownMs = {};
let patienceFrames = {};
let warmupMs = 1500;
let defaultMelHop = 8;
let lastHitMs = {};
let aboveThresholdFrames = {};
let peakScores = {};
let startupRequireArming = true;
let startupArmCeiling = 0.9;
let startupArmFrames = 2;
let armedLabels = {};
let belowArmFrames = {};
let sessionStartMs = 0;
let inferenceWindows = 0;

function mapByLabels(valueFactory) {
  const out = {};
  for (const label of wakewordLabels) {
    out[label] = valueFactory(label);
  }
  return out;
}

function defaultThresholdForLabel(_label) {
  return 0.6;
}

function defaultPatienceForLabel(_label) {
  return 2;
}

function defaultCooldownForLabel(_label) {
  return 1000;
}

function postError(code, message, detail = null) {
  self.postMessage({
    type: 'error',
    payload: { code, message, detail },
  });
}

function resetState() {
  melBuffer = [];
  embeddingBuffer = [];
  inferenceWindows = 0;
  sessionStartMs = Date.now();
  aboveThresholdFrames = mapByLabels(() => 0);
  peakScores = mapByLabels(() => 0);
  armedLabels = mapByLabels(() => !startupRequireArming);
  belowArmFrames = mapByLabels(() => 0);
  for (let i = 0; i < 16; i += 1) {
    embeddingBuffer.push(new Float32Array(96));
  }
}

function applyMelTransform(source) {
  const out = new Float32Array(source.length);
  for (let i = 0; i < source.length; i += 1) {
    out[i] = source[i] / 10.0 + 2.0;
  }
  return out;
}

function maybeTransformMel(source, options = {}) {
  if (options.melTransform === false) {
    return source;
  }
  return applyMelTransform(source);
}

async function initSessions(config) {
  if (!self.ort) {
    throw new Error(
      `Failed to load onnxruntime-web script. Last error: ${ortScriptLoadError?.message || 'unknown'}`
    );
  }

  ort.env.wasm.wasmPaths = '/vendor/ort/';
  const sessionOptions = { executionProviders: ['wasm'] };

  melspecSession = await ort.InferenceSession.create(config.melspecModelUrl, sessionOptions);
  embeddingSession = await ort.InferenceSession.create(config.embeddingModelUrl, sessionOptions);

  const wakewordModelUrls = config.wakewordModelUrls || {};
  wakewordSessions = {};
  for (const [label, modelUrl] of Object.entries(wakewordModelUrls)) {
    wakewordSessions[label] = await ort.InferenceSession.create(modelUrl, sessionOptions);
  }

  wakewordLabels = Object.keys(wakewordSessions);
  if (wakewordLabels.length === 0) {
    throw new Error('No wakeword models configured');
  }

  thresholds = {
    ...mapByLabels((label) => defaultThresholdForLabel(label)),
    ...(config.thresholds || {}),
  };
  cooldownMs = {
    ...mapByLabels((label) => defaultCooldownForLabel(label)),
    ...(config.cooldownMs || {}),
  };
  patienceFrames = {
    ...mapByLabels((label) => defaultPatienceForLabel(label)),
    ...(config.patienceFrames || {}),
  };
  lastHitMs = mapByLabels(() => 0);
  warmupMs = Number.isFinite(config.warmupMs) ? config.warmupMs : warmupMs;
  defaultMelHop = Number.isFinite(config.melHop) ? Math.max(1, Number(config.melHop)) : defaultMelHop;
  startupRequireArming = config.startupRequireArming !== false;
  startupArmCeiling = Number.isFinite(config.startupArmCeiling)
    ? Number(config.startupArmCeiling)
    : startupArmCeiling;
  startupArmFrames = Number.isFinite(config.startupArmFrames)
    ? Math.max(1, Number(config.startupArmFrames))
    : startupArmFrames;
  resetState();
}

async function runInference(chunk, emitEvents = true, options = {}) {
  if (chunk.length !== FRAME_SIZE) {
    return;
  }

  const melHop = Math.max(1, Number(options.melHop || defaultMelHop));

  const melspecInput = new ort.Tensor('float32', chunk, [1, FRAME_SIZE]);
  const melspecOutputMap = await melspecSession.run({ [melspecSession.inputNames[0]]: melspecInput });
  const melRaw = melspecOutputMap[melspecSession.outputNames[0]].data;
  const melData = maybeTransformMel(melRaw, options);

  for (let i = 0; i < 5; i += 1) {
    melBuffer.push(new Float32Array(melData.subarray(i * 32, (i + 1) * 32)));
  }

  while (melBuffer.length >= 76) {
    const flattenedMel = new Float32Array(76 * 32);
    for (let i = 0; i < 76; i += 1) {
      flattenedMel.set(melBuffer[i], i * 32);
    }

    const embeddingInput = new ort.Tensor('float32', flattenedMel, [1, 76, 32, 1]);
    const embeddingOutputMap = await embeddingSession.run({ [embeddingSession.inputNames[0]]: embeddingInput });
    const embeddingRaw = embeddingOutputMap[embeddingSession.outputNames[0]].data;

    embeddingBuffer.shift();
    embeddingBuffer.push(new Float32Array(embeddingRaw));

    const flattenedEmbeddings = new Float32Array(16 * 96);
    for (let i = 0; i < 16; i += 1) {
      flattenedEmbeddings.set(embeddingBuffer[i], i * 96);
    }

    const wakewordInput = new ort.Tensor('float32', flattenedEmbeddings, [1, 16, 96]);

    const scorePayload = mapByLabels(() => 0);
    const nowMs = Date.now();
    const warmedUp = nowMs - sessionStartMs >= warmupMs;
    const abovePayload = mapByLabels(() => false);

    for (const [label, session] of Object.entries(wakewordSessions)) {
      const outputMap = await session.run({ [session.inputNames[0]]: wakewordInput });
      const score = Number(outputMap[session.outputNames[0]].data[0] || 0);
      scorePayload[label] = score;
      if (score > peakScores[label]) {
        peakScores[label] = score;
      }

      if (emitEvents && score >= 0.35) {
        self.postMessage({
          type: 'trace',
          payload: {
            ts: nowMs,
            label,
            score,
            threshold: thresholds[label] ?? null,
            aboveThresholdFrames: aboveThresholdFrames[label] ?? 0,
            warmedUp,
            armed: Boolean(armedLabels[label]),
          },
        });
      }

      if (score > (thresholds[label] ?? Number.POSITIVE_INFINITY)) {
        aboveThresholdFrames[label] += 1;
        abovePayload[label] = true;
      } else {
        aboveThresholdFrames[label] = 0;
      }

      if (!armedLabels[label]) {
        if (score < startupArmCeiling) {
          belowArmFrames[label] += 1;
          if (belowArmFrames[label] >= startupArmFrames) {
            armedLabels[label] = true;
          }
        } else {
          belowArmFrames[label] = 0;
        }
        aboveThresholdFrames[label] = 0;
        continue;
      }

      const patienceSatisfied = aboveThresholdFrames[label] >= (patienceFrames[label] ?? 1);
      if (
        emitEvents &&
        warmedUp &&
        patienceSatisfied &&
        nowMs - lastHitMs[label] >= (cooldownMs[label] ?? 0)
      ) {
        lastHitMs[label] = nowMs;
        self.postMessage({
          type: 'hit',
          payload: { label, score, ts: nowMs },
        });
      }
    }

    if (emitEvents) {
      self.postMessage({
        type: 'score',
        payload: {
          ts: nowMs,
          inferenceWindows,
          scores: scorePayload,
          peaks: { ...peakScores },
        },
      });
    }

    inferenceWindows += 1;

    if (emitEvents && inferenceWindows % 5 === 0) {
      self.postMessage({
        type: 'debug',
        payload: {
          ts: nowMs,
          warmedUp,
          warmupRemainingMs: Math.max(0, warmupMs - (nowMs - sessionStartMs)),
          inferenceWindows,
          melBufferLen: melBuffer.length,
          aboveThresholdFrames: { ...aboveThresholdFrames },
          armed: { ...armedLabels },
          aboveThreshold: abovePayload,
          thresholds: { ...thresholds },
          peaks: { ...peakScores },
          scores: scorePayload,
        },
      });
    }

    melBuffer.splice(0, melHop);
  }
}

self.onmessage = async (event) => {
  try {
    const { type, payload } = event.data || {};

    if (type === 'init') {
      await initSessions(payload || {});
      self.postMessage({ type: 'ready', payload: { sampleRate: 16000 } });
      return;
    }

    if (type === 'reset') {
      resetState();
      return;
    }

    if (type === 'frame') {
      if (!melspecSession || !embeddingSession) {
        return;
      }
      const frame = payload?.chunk;
      if (!(frame instanceof Float32Array)) {
        return;
      }
      await runInference(frame);
    }
  } catch (error) {
    postError('WAKEWORD_WORKER_ERROR', error?.message || String(error), {
      stack: error?.stack || null,
    });
  }
};
