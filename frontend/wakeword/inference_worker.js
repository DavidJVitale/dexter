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

let melBuffer = [];
let embeddingBuffer = [];
let thresholds = {
  dexter_start: 0.56,
  dexter_stop: 0.66,
  dexter_abort: 0.66,
};
let cooldownMs = {
  dexter_start: 1000,
  dexter_stop: 1000,
  dexter_abort: 1000,
};
let patienceFrames = {
  dexter_start: 2,
  dexter_stop: 2,
  dexter_abort: 2,
};
let warmupMs = 1500;
let lastHitMs = {
  dexter_start: 0,
  dexter_stop: 0,
  dexter_abort: 0,
};
let aboveThresholdFrames = {
  dexter_start: 0,
  dexter_stop: 0,
  dexter_abort: 0,
};
let peakScores = {
  dexter_start: 0,
  dexter_stop: 0,
  dexter_abort: 0,
};
let sessionStartMs = 0;
let inferenceWindows = 0;

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
  aboveThresholdFrames = {
    dexter_start: 0,
    dexter_stop: 0,
    dexter_abort: 0,
  };
  peakScores = {
    dexter_start: 0,
    dexter_stop: 0,
    dexter_abort: 0,
  };
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

function expectedLabelFromFilename(name) {
  const lower = String(name || '').toLowerCase();
  if (lower.includes('start')) return 'dexter_start';
  if (lower.includes('stop')) return 'dexter_stop';
  if (lower.includes('abort')) return 'dexter_abort';
  return 'unrelated';
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

  wakewordSessions = {
    dexter_start: await ort.InferenceSession.create(config.wakewordModelUrls.dexter_start, sessionOptions),
    dexter_stop: await ort.InferenceSession.create(config.wakewordModelUrls.dexter_stop, sessionOptions),
    dexter_abort: await ort.InferenceSession.create(config.wakewordModelUrls.dexter_abort, sessionOptions),
  };

  thresholds = { ...thresholds, ...(config.thresholds || {}) };
  cooldownMs = { ...cooldownMs, ...(config.cooldownMs || {}) };
  patienceFrames = { ...patienceFrames, ...(config.patienceFrames || {}) };
  warmupMs = Number.isFinite(config.warmupMs) ? config.warmupMs : warmupMs;
  resetState();
}

async function runInference(chunk, emitEvents = true) {
  if (chunk.length !== FRAME_SIZE) {
    return;
  }

  const melspecInput = new ort.Tensor('float32', chunk, [1, FRAME_SIZE]);
  const melspecOutputMap = await melspecSession.run({ [melspecSession.inputNames[0]]: melspecInput });
  const melRaw = melspecOutputMap[melspecSession.outputNames[0]].data;
  const melData = applyMelTransform(melRaw);

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

    const scorePayload = {
      dexter_start: 0,
      dexter_stop: 0,
      dexter_abort: 0,
    };
    const nowMs = Date.now();
    const warmedUp = nowMs - sessionStartMs >= warmupMs;
    const abovePayload = {
      dexter_start: false,
      dexter_stop: false,
      dexter_abort: false,
    };

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
            threshold: thresholds[label],
            aboveThresholdFrames: aboveThresholdFrames[label],
            warmedUp,
          },
        });
      }

      if (score > thresholds[label]) {
        aboveThresholdFrames[label] += 1;
        abovePayload[label] = true;
      } else {
        aboveThresholdFrames[label] = 0;
      }

      const patienceSatisfied = aboveThresholdFrames[label] >= patienceFrames[label];
      if (
        emitEvents &&
        warmedUp &&
        patienceSatisfied &&
        nowMs - lastHitMs[label] >= cooldownMs[label]
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
          aboveThreshold: abovePayload,
          thresholds: { ...thresholds },
          peaks: { ...peakScores },
          scores: scorePayload,
        },
      });
    }

    melBuffer.splice(0, 8);
  }
}

async function runBenchmarkSamples(files) {
  const results = [];

  for (const file of files) {
    const samples = file.samples;
    if (!(samples instanceof Float32Array)) {
      throw new Error(`benchmark sample payload missing Float32Array for file=${file.name}`);
    }
    resetState();

    for (let i = 0; i < samples.length; i += FRAME_SIZE) {
      const chunk = new Float32Array(FRAME_SIZE);
      const slice = samples.subarray(i, Math.min(i + FRAME_SIZE, samples.length));
      chunk.set(slice, 0);
      await runInference(chunk, false);
    }

    const peaks = { ...peakScores };
    const winner = Object.entries(peaks).reduce((a, b) => (a[1] >= b[1] ? a : b))[0];
    results.push({
      file: file.name,
      expected: expectedLabelFromFilename(file.name),
      sampleRateInput: Number(file.sampleRateInput || 16000),
      sampleRateRuntime: Number(file.sampleRateRuntime || 16000),
      chunkSamples: FRAME_SIZE,
      durationSec16k: Number((samples.length / 16000).toFixed(4)),
      winnerByPeak: winner,
      scores: {
        dexter_start: { peak: peaks.dexter_start },
        dexter_stop: { peak: peaks.dexter_stop },
        dexter_abort: { peak: peaks.dexter_abort },
      },
    });
  }

  self.postMessage({
    type: 'benchmark_result',
    payload: {
      benchmark: 'openwakeword_wasm_upload_benchmark_v1',
      generatedAtEpoch: Math.floor(Date.now() / 1000),
      files: results,
    },
  });
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
      return;
    }

    if (type === 'benchmark_samples') {
      if (!melspecSession || !embeddingSession) {
        return;
      }
      const files = payload?.files;
      if (!Array.isArray(files) || files.length === 0) {
        throw new Error('benchmark_samples payload requires a non-empty files array');
      }
      await runBenchmarkSamples(files);
    }
  } catch (error) {
    postError('WAKEWORD_WORKER_ERROR', error?.message || String(error), {
      stack: error?.stack || null,
    });
  }
};
