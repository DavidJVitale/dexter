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
let wakewordLabels = ['dexter_start', 'dexter_stop', 'dexter_abort'];

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
let sessionStartMs = 0;
let inferenceWindows = 0;

function mapByLabels(valueFactory) {
  const out = {};
  for (const label of wakewordLabels) {
    out[label] = valueFactory(label);
  }
  return out;
}

function defaultThresholdForLabel(label) {
  if (label === 'dexter_start') return 0.56;
  if (label === 'dexter_stop') return 0.66;
  if (label === 'dexter_abort') return 0.66;
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

function expectedLabelFromFilename(name) {
  const lower = String(name || '').toLowerCase();
  if (lower.includes('start')) return 'dexter_start';
  if (lower.includes('stop')) return 'dexter_stop';
  if (lower.includes('abort')) return 'dexter_abort';
  return 'unrelated';
}

function summarizeSeries(values) {
  if (!values.length) {
    return { peak: 0, mean: 0, p95: 0, peakFrame: -1, n: 0 };
  }
  let peak = values[0];
  let peakFrame = 0;
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    sum += v;
    if (v > peak) {
      peak = v;
      peakFrame = i;
    }
  }
  const sorted = Array.from(values).sort((a, b) => a - b);
  const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
  return {
    peak,
    mean: sum / values.length,
    p95,
    peakFrame,
    n: values.length,
  };
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
  resetState();
}

async function runInference(chunk, emitEvents = true, options = {}) {
  if (chunk.length !== FRAME_SIZE) {
    return [];
  }
  const windowScores = [];
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
          },
        });
      }

      if (score > (thresholds[label] ?? Number.POSITIVE_INFINITY)) {
        aboveThresholdFrames[label] += 1;
        abovePayload[label] = true;
      } else {
        aboveThresholdFrames[label] = 0;
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
    windowScores.push({
      dexter_start: scorePayload.dexter_start,
      dexter_stop: scorePayload.dexter_stop,
      dexter_abort: scorePayload.dexter_abort,
    });
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

    melBuffer.splice(0, melHop);
  }
  return windowScores;
}

async function runBenchmarkSamples(files) {
  const results = [];

  for (const file of files) {
    const variants = Array.isArray(file.variants) ? file.variants : [file];
    const variantResults = [];

    for (const variant of variants) {
      const samples = variant.samples;
      if (!(samples instanceof Float32Array)) {
        throw new Error(`benchmark sample payload missing Float32Array for file=${file.name}`);
      }
      const trace = {
        dexter_start: [],
        dexter_stop: [],
        dexter_abort: [],
      };

      const sampleOffset = Number(variant.sampleOffset || 0);
      const frameStep = Math.max(1, Number(variant.inferenceOptions?.frameStep || FRAME_SIZE));
      resetState();
      for (let i = sampleOffset; i < samples.length; i += frameStep) {
        const chunk = new Float32Array(FRAME_SIZE);
        const slice = samples.subarray(i, Math.min(i + FRAME_SIZE, samples.length));
        chunk.set(slice, 0);
        const chunkScores = await runInference(chunk, false, variant.inferenceOptions || {});
        for (const score of chunkScores) {
          trace.dexter_start.push(score.dexter_start);
          trace.dexter_stop.push(score.dexter_stop);
          trace.dexter_abort.push(score.dexter_abort);
        }
      }

      const peaks = { ...peakScores };
      const winner = Object.entries(peaks).reduce((a, b) => (a[1] >= b[1] ? a : b))[0];
      variantResults.push({
        variant: variant.variant || 'unknown',
        sampleRateInput: Number(variant.sampleRateInput || 16000),
        sampleRateRuntime: Number(variant.sampleRateRuntime || 16000),
        durationSec16k: Number((samples.length / 16000).toFixed(4)),
        frameStep,
        signalStats: variant.signalStats || null,
        winnerByPeak: winner,
        scores: {
          dexter_start: summarizeSeries(trace.dexter_start),
          dexter_stop: summarizeSeries(trace.dexter_stop),
          dexter_abort: summarizeSeries(trace.dexter_abort),
        },
        trace:
          String(file.name || '').toLowerCase().includes('stop2')
            ? {
                dexter_start: trace.dexter_start,
                dexter_stop: trace.dexter_stop,
                dexter_abort: trace.dexter_abort,
              }
            : undefined,
      });
    }

    let offsetSweep = null;
    let melHopSweep = null;
    const fileNameLower = String(file.name || '').toLowerCase();
    if (fileNameLower.includes('dexter_stop2')) {
      const baseVariant =
        variantResults.find((v) => v.variant === 'js_prepared_mel_on') ||
        variantResults[0];
      const sourceVariant = variants.find((v) => v.variant === baseVariant?.variant);
      if (sourceVariant?.samples instanceof Float32Array) {
        const offsets = [0, 160, 320, 480, 640, 800, 960, 1120];
        const rows = [];
        for (const sampleOffset of offsets) {
          resetState();
          const trace = [];
          for (let i = sampleOffset; i < sourceVariant.samples.length; i += FRAME_SIZE) {
            const chunk = new Float32Array(FRAME_SIZE);
            const slice = sourceVariant.samples.subarray(
              i,
              Math.min(i + FRAME_SIZE, sourceVariant.samples.length)
            );
            chunk.set(slice, 0);
            const chunkScores = await runInference(chunk, false, sourceVariant.inferenceOptions || {});
            for (const score of chunkScores) {
              trace.push(score.dexter_stop);
            }
          }
          const stopStats = summarizeSeries(trace);
          rows.push({
            sampleOffset,
            stopPeak: stopStats.peak,
            stopMean: stopStats.mean,
            stopPeakFrame: stopStats.peakFrame,
            n: stopStats.n,
          });
        }
        const best = rows.reduce((acc, row) => (row.stopPeak > acc.stopPeak ? row : acc), rows[0]);
        offsetSweep = {
          variant: sourceVariant.variant,
          offsets: rows,
          best,
        };

        const hops = [1, 2, 3, 4, 5, 6, 7, 8];
        const hopRows = [];
        for (const melHop of hops) {
          resetState();
          const stopTrace = [];
          for (let i = 0; i < sourceVariant.samples.length; i += FRAME_SIZE) {
            const chunk = new Float32Array(FRAME_SIZE);
            const slice = sourceVariant.samples.subarray(
              i,
              Math.min(i + FRAME_SIZE, sourceVariant.samples.length)
            );
            chunk.set(slice, 0);
            const chunkScores = await runInference(chunk, false, {
              ...(sourceVariant.inferenceOptions || {}),
              melHop,
              frameStep: FRAME_SIZE,
            });
            for (const score of chunkScores) {
              stopTrace.push(score.dexter_stop);
            }
          }
          const stopStats = summarizeSeries(stopTrace);
          hopRows.push({
            melHop,
            stopPeak: stopStats.peak,
            stopMean: stopStats.mean,
            stopPeakFrame: stopStats.peakFrame,
            n: stopStats.n,
          });
        }
        const bestHop = hopRows.reduce((acc, row) => (row.stopPeak > acc.stopPeak ? row : acc), hopRows[0]);
        melHopSweep = {
          variant: sourceVariant.variant,
          hops: hopRows,
          best: bestHop,
        };
      }
    }

    let diagnostics = null;
    const jsPrepared = variantResults.find((v) => v.variant === 'js_prepared_mel_on');
    const pythonGolden = variantResults.find((v) => v.variant === 'python_golden_pcm16_mel_on');
    if (jsPrepared && pythonGolden) {
      diagnostics = {
        stopPeakDelta: jsPrepared.scores.dexter_stop.peak - pythonGolden.scores.dexter_stop.peak,
        startPeakDelta: jsPrepared.scores.dexter_start.peak - pythonGolden.scores.dexter_start.peak,
        abortPeakDelta: jsPrepared.scores.dexter_abort.peak - pythonGolden.scores.dexter_abort.peak,
        stopPeakRatio:
          pythonGolden.scores.dexter_stop.peak > 0
            ? jsPrepared.scores.dexter_stop.peak / pythonGolden.scores.dexter_stop.peak
            : null,
      };
    }

    let overlapDiagnostics = null;
    const baseStep = variantResults.find((v) => v.variant === 'js_prepared_mel_on');
    const denseStep = variantResults.find((v) => v.variant === 'js_prepared_mel_on_step160');
    if (baseStep && denseStep) {
      overlapDiagnostics = {
        jsStopPeakStep1280: baseStep.scores.dexter_stop.peak,
        jsStopPeakStep160: denseStep.scores.dexter_stop.peak,
        jsStopPeakRatioStep160Over1280:
          baseStep.scores.dexter_stop.peak > 0
            ? denseStep.scores.dexter_stop.peak / baseStep.scores.dexter_stop.peak
            : null,
        jsStartPeakStep1280: baseStep.scores.dexter_start.peak,
        jsStartPeakStep160: denseStep.scores.dexter_start.peak,
      };
    }

    results.push({
      file: file.name,
      expected: file.expected || expectedLabelFromFilename(file.name),
      chunkSamples: FRAME_SIZE,
      variants: variantResults,
      diagnostics,
      overlapDiagnostics,
      offsetSweep,
      melHopSweep,
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
