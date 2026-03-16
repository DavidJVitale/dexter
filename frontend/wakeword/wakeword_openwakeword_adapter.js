import { WakewordAudioCaptureController } from './audio_capture_controller.js';

const DEFAULT_BACKEND_ORIGIN =
  window.location.port === '5173'
    ? `${window.location.protocol}//${window.location.hostname}:5050`
    : window.location.origin;
const PRIMARY_WAKEWORD_MODEL_PATHS = {
  dexter: '/models/wakewords/dexter.onnx',
  alexa_v0_1: '/models/openwakeword_resources/alexa_v0.1.onnx',
  hey_jarvis_v0_1: '/models/openwakeword_resources/hey_jarvis_v0.1.onnx',
};

export class OpenWakeWordAdapter {
  constructor(config = {}) {
    this.config = config;
    this.listeners = new Map();
    this.audio = new WakewordAudioCaptureController();
    this.worker = null;
    this.initialized = false;
    this.running = false;
    this.readyPromise = null;
    this.readyResolve = null;
    this.readyReject = null;
    this.decodeAudioContext = null;
    this.goldenVectorsPromise = null;
    this.goldenVectorsByName = null;
  }

  on(eventName, callback) {
    if (!this.listeners.has(eventName)) {
      this.listeners.set(eventName, new Set());
    }
    const set = this.listeners.get(eventName);
    set.add(callback);
    return () => set.delete(callback);
  }

  emit(eventName, payload) {
    const callbacks = this.listeners.get(eventName);
    if (!callbacks) {
      return;
    }
    for (const callback of callbacks) {
      callback(payload);
    }
  }

  async initialize() {
    if (this.initialized) {
      return;
    }

    const backendOrigin = this.config.backendOrigin || DEFAULT_BACKEND_ORIGIN;

    this.readyPromise = new Promise((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });

    this.worker = new Worker(new URL('./inference_worker.js', import.meta.url));
    this.worker.onmessage = (event) => {
      const { type, payload } = event.data || {};
      if (!type) {
        return;
      }
      if (type === 'ready' && this.readyResolve) {
        this.readyResolve(payload);
      }
      if (type === 'error' && this.readyReject) {
        this.readyReject(new Error(payload?.message || 'wakeword worker init failed'));
      }
      this.emit(type, payload);
    };

    const wakewordModelUrls = Object.fromEntries(
      Object.entries({
        ...PRIMARY_WAKEWORD_MODEL_PATHS,
        ...(this.config.wakewordModelUrls || {}),
      }).map(([label, pathOrUrl]) => {
        const asString = String(pathOrUrl || "");
        const isAbsolute = /^https?:\/\//i.test(asString);
        return [label, isAbsolute ? asString : `${backendOrigin}${asString}`];
      })
    );
    const defaultThresholds = {
      dexter: 0.6,
      alexa_v0_1: 0.6,
      hey_jarvis_v0_1: 0.6,
    };
    const defaultCooldownMs = {
      dexter: 1000,
      alexa_v0_1: 1000,
      hey_jarvis_v0_1: 1000,
    };
    const defaultPatienceFrames = {
      dexter: 2,
      alexa_v0_1: 2,
      hey_jarvis_v0_1: 2,
    };

    this.worker.postMessage({
      type: 'init',
      payload: {
        melspecModelUrl: `${backendOrigin}/models/openwakeword_resources/melspectrogram.onnx`,
        embeddingModelUrl: `${backendOrigin}/models/openwakeword_resources/embedding_model.onnx`,
        wakewordModelUrls,
        thresholds: { ...defaultThresholds, ...(this.config.thresholds || {}) },
        cooldownMs: { ...defaultCooldownMs, ...(this.config.cooldownMs || {}) },
        patienceFrames: { ...defaultPatienceFrames, ...(this.config.patienceFrames || {}) },
        warmupMs: 1500,
        melHop: 5,
      },
    });

    await this.readyPromise;
    this.initialized = true;
  }

  async start() {
    if (this.running) {
      return;
    }
    if (!this.initialized) {
      await this.initialize();
    }

    await this.audio.start((chunk) => {
      if (!this.worker) {
        return;
      }
      this.worker.postMessage({ type: 'frame', payload: { chunk } }, [chunk.buffer]);
    });

    this.running = true;
  }

  async stop() {
    if (!this.running) {
      return;
    }
    await this.audio.stop();
    this.running = false;
  }

  async runWavBenchmark(files) {
    if (!this.initialized) {
      await this.initialize();
    }
    if (!this.worker) {
      throw new Error('wakeword worker is not available');
    }

    const prepared = [];
    const transferList = [];
    const goldenByName = await this.loadGoldenVectorsByName();
    for (const file of files) {
      const arrayBuffer = await file.arrayBuffer();
      const decoded = await this.decodeFileTo16kMonoFloat32(file.name, arrayBuffer);
      const variants = [
        {
          variant: "js_prepared_mel_on",
          sampleRateInput: decoded.sampleRateInput,
          sampleRateRuntime: decoded.sampleRateRuntime,
          samples: decoded.samples,
          signalStats: this.computeSignalStats(decoded.samples, decoded.sampleRateRuntime),
          inferenceOptions: { melTransform: true, trimLeadingSilence: false, frameStep: 1280 },
        },
        {
          variant: "js_prepared_mel_off",
          sampleRateInput: decoded.sampleRateInput,
          sampleRateRuntime: decoded.sampleRateRuntime,
          samples: decoded.samples.slice(0),
          signalStats: this.computeSignalStats(decoded.samples, decoded.sampleRateRuntime),
          inferenceOptions: { melTransform: false, trimLeadingSilence: false, frameStep: 1280 },
        },
        {
          variant: "js_prepared_mel_on_step160",
          sampleRateInput: decoded.sampleRateInput,
          sampleRateRuntime: decoded.sampleRateRuntime,
          samples: decoded.samples.slice(0),
          signalStats: this.computeSignalStats(decoded.samples, decoded.sampleRateRuntime),
          inferenceOptions: { melTransform: true, trimLeadingSilence: false, frameStep: 160 },
        },
      ];
      transferList.push(decoded.samples.buffer);
      transferList.push(variants[1].samples.buffer);
      transferList.push(variants[2].samples.buffer);

      const trimmed = this.trimLeadingSilencePcm16Float(decoded.samples, 1000);
      if (trimmed.trimmedSampleCount > 0) {
        variants.push({
          variant: "js_prepared_mel_on_trim",
          sampleRateInput: decoded.sampleRateInput,
          sampleRateRuntime: decoded.sampleRateRuntime,
          samples: trimmed.samples,
          signalStats: this.computeSignalStats(trimmed.samples, decoded.sampleRateRuntime),
          inferenceOptions: { melTransform: true, trimLeadingSilence: true, frameStep: 1280 },
        });
      }
      if (trimmed.trimmedSampleCount > 0) {
        transferList.push(trimmed.samples.buffer);
      }

      const golden = goldenByName?.[file.name];
      if (golden) {
        const goldenSamples = this.decodeBase64Int16ToFloat32(golden.pcm16LeBase64);
        variants.push({
          variant: "python_golden_pcm16_mel_on",
          sampleRateInput: golden.sampleRateInput || 16000,
          sampleRateRuntime: golden.sampleRateRuntime || 16000,
          samples: goldenSamples,
          signalStats: this.computeSignalStats(goldenSamples, golden.sampleRateRuntime || 16000),
          inferenceOptions: { melTransform: true, trimLeadingSilence: false, frameStep: 1280 },
        });
        transferList.push(goldenSamples.buffer);
        variants.push({
          variant: "python_golden_pcm16_mel_off",
          sampleRateInput: golden.sampleRateInput || 16000,
          sampleRateRuntime: golden.sampleRateRuntime || 16000,
          samples: goldenSamples.slice(0),
          signalStats: this.computeSignalStats(goldenSamples, golden.sampleRateRuntime || 16000),
          inferenceOptions: { melTransform: false, trimLeadingSilence: false, frameStep: 1280 },
        });
        transferList.push(variants[variants.length - 1].samples.buffer);
        variants.push({
          variant: "python_golden_pcm16_mel_on_step160",
          sampleRateInput: golden.sampleRateInput || 16000,
          sampleRateRuntime: golden.sampleRateRuntime || 16000,
          samples: goldenSamples.slice(0),
          signalStats: this.computeSignalStats(goldenSamples, golden.sampleRateRuntime || 16000),
          inferenceOptions: { melTransform: true, trimLeadingSilence: false, frameStep: 160 },
        });
        transferList.push(variants[variants.length - 1].samples.buffer);
      }

      prepared.push({
        name: file.name,
        expected: this.expectedLabelFromFilename(file.name),
        variants,
      });
    }

    this.worker.postMessage(
      {
        type: 'benchmark_samples',
        payload: {
          files: prepared,
        },
      },
      transferList
    );
  }

  async getDecodeAudioContext() {
    if (this.decodeAudioContext) {
      return this.decodeAudioContext;
    }
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    this.decodeAudioContext = new AudioCtx();
    if (this.decodeAudioContext.state === 'suspended') {
      try {
        await this.decodeAudioContext.resume();
      } catch (_error) {
        // Ignore; decodeAudioData may still work while suspended.
      }
    }
    return this.decodeAudioContext;
  }

  expectedLabelFromFilename(name) {
    const lower = String(name || "").toLowerCase();
    if (lower.includes("start")) return "dexter_start";
    if (lower.includes("stop")) return "dexter_stop";
    if (lower.includes("abort")) return "dexter_abort";
    return "unrelated";
  }

  async loadGoldenVectorsByName() {
    if (this.goldenVectorsByName) {
      return this.goldenVectorsByName;
    }
    if (!this.goldenVectorsPromise) {
      this.goldenVectorsPromise = fetch("/wakeword/tests/wakeword_golden_pcm16.json")
        .then((response) => {
          if (!response.ok) {
            return null;
          }
          return response.json();
        })
        .then((payload) => {
          if (!payload?.files || !Array.isArray(payload.files)) {
            return {};
          }
          const byName = {};
          for (const item of payload.files) {
            if (item?.name) {
              byName[item.name] = item;
            }
          }
          return byName;
        })
        .catch(() => ({}));
    }
    this.goldenVectorsByName = await this.goldenVectorsPromise;
    return this.goldenVectorsByName;
  }

  decodeBase64Int16ToFloat32(base64) {
    const raw = atob(base64);
    const byteLen = raw.length;
    const bytes = new Uint8Array(byteLen);
    for (let i = 0; i < byteLen; i += 1) {
      bytes[i] = raw.charCodeAt(i);
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const sampleCount = Math.floor(view.byteLength / 2);
    const out = new Float32Array(sampleCount);
    for (let i = 0; i < sampleCount; i += 1) {
      out[i] = view.getInt16(i * 2, true);
    }
    return out;
  }

  computeSignalStats(samples, sampleRate) {
    if (!samples || samples.length === 0) {
      return {
        sampleCount: 0,
        durationSec: 0,
        rms: 0,
        peakAbs: 0,
        p95Abs: 0,
        leadingSilenceSecThr1000: 0,
      };
    }
    let sumSq = 0;
    let peakAbs = 0;
    const abs = new Float32Array(samples.length);
    let firstAbove = -1;
    for (let i = 0; i < samples.length; i += 1) {
      const v = samples[i];
      sumSq += v * v;
      const a = Math.abs(v);
      abs[i] = a;
      if (a > peakAbs) peakAbs = a;
      if (firstAbove < 0 && a >= 1000) {
        firstAbove = i;
      }
    }
    abs.sort();
    const p95 = abs[Math.min(abs.length - 1, Math.floor(abs.length * 0.95))];
    return {
      sampleCount: samples.length,
      durationSec: Number((samples.length / sampleRate).toFixed(4)),
      rms: Math.sqrt(sumSq / samples.length),
      peakAbs,
      p95Abs: p95,
      leadingSilenceSecThr1000:
        firstAbove >= 0 ? Number((firstAbove / sampleRate).toFixed(4)) : Number((samples.length / sampleRate).toFixed(4)),
    };
  }

  trimLeadingSilencePcm16Float(samples, thresholdAbs) {
    let firstAbove = -1;
    for (let i = 0; i < samples.length; i += 1) {
      if (Math.abs(samples[i]) >= thresholdAbs) {
        firstAbove = i;
        break;
      }
    }
    if (firstAbove <= 0) {
      return { samples: samples.slice(0), trimmedSampleCount: 0 };
    }
    return {
      samples: samples.slice(firstAbove),
      trimmedSampleCount: firstAbove,
    };
  }

  async decodeFileTo16kMonoFloat32(_name, arrayBuffer) {
    let srcRate = 16000;
    let mono;
    try {
      const parsed = this.parseWavPcmToMonoFloat32(arrayBuffer);
      srcRate = parsed.sampleRate;
      mono = parsed.samples;
    } catch (_parseError) {
      const decodeCtx = await this.getDecodeAudioContext();
      const decodedBuffer = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
      srcRate = decodedBuffer.sampleRate;
      const channels = decodedBuffer.numberOfChannels;
      const length = decodedBuffer.length;

      mono = new Float32Array(length);
      for (let ch = 0; ch < channels; ch += 1) {
        const channelData = decodedBuffer.getChannelData(ch);
        for (let i = 0; i < length; i += 1) {
          mono[i] += channelData[i] / channels;
        }
      }
    }

    if (srcRate === 16000) {
      const pcm16Scale = this.toPcm16FloatScale(mono);
      return {
        sampleRateInput: srcRate,
        sampleRateRuntime: 16000,
        samples: pcm16Scale,
      };
    }

    const resampled = this.linearResample(mono, srcRate, 16000);
    const pcm16Scale = this.toPcm16FloatScale(resampled);
    return {
      sampleRateInput: srcRate,
      sampleRateRuntime: 16000,
      samples: pcm16Scale,
    };
  }

  parseWavPcmToMonoFloat32(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    if (view.byteLength < 44) {
      throw new Error('WAV too small');
    }

    const riff = this.readFourCC(view, 0);
    const wave = this.readFourCC(view, 8);
    if (riff !== 'RIFF' || wave !== 'WAVE') {
      throw new Error('Unsupported WAV container');
    }

    let offset = 12;
    let audioFormat = 1;
    let channels = 1;
    let sampleRate = 16000;
    let bitsPerSample = 16;
    let dataOffset = -1;
    let dataSize = 0;

    while (offset + 8 <= view.byteLength) {
      const chunkId = this.readFourCC(view, offset);
      const chunkSize = view.getUint32(offset + 4, true);
      const chunkDataOffset = offset + 8;

      if (chunkId === 'fmt ') {
        audioFormat = view.getUint16(chunkDataOffset, true);
        channels = view.getUint16(chunkDataOffset + 2, true);
        sampleRate = view.getUint32(chunkDataOffset + 4, true);
        bitsPerSample = view.getUint16(chunkDataOffset + 14, true);
      } else if (chunkId === 'data') {
        dataOffset = chunkDataOffset;
        dataSize = chunkSize;
        break;
      }

      offset = chunkDataOffset + chunkSize + (chunkSize % 2);
    }

    if (dataOffset < 0) {
      throw new Error('WAV missing data chunk');
    }

    // WAVE_FORMAT_EXTENSIBLE
    if (audioFormat === 65534) {
      audioFormat = 1;
    }

    const bytesPerSample = bitsPerSample / 8;
    if (!Number.isInteger(bytesPerSample) || bytesPerSample <= 0) {
      throw new Error('Invalid WAV sample width');
    }

    const frameCount = Math.floor(dataSize / (bytesPerSample * channels));
    const mono = new Float32Array(frameCount);

    for (let i = 0; i < frameCount; i += 1) {
      let sum = 0;
      for (let ch = 0; ch < channels; ch += 1) {
        const sampleOffset = dataOffset + (i * channels + ch) * bytesPerSample;
        let sample = 0;
        if (audioFormat === 1 && bitsPerSample === 16) {
          sample = view.getInt16(sampleOffset, true) / 32768;
        } else if (audioFormat === 1 && bitsPerSample === 8) {
          sample = (view.getUint8(sampleOffset) - 128) / 128;
        } else if (audioFormat === 1 && bitsPerSample === 32) {
          sample = view.getInt32(sampleOffset, true) / 2147483648;
        } else if (audioFormat === 3 && bitsPerSample === 32) {
          sample = view.getFloat32(sampleOffset, true);
        } else {
          throw new Error(`Unsupported WAV format: audioFormat=${audioFormat} bits=${bitsPerSample}`);
        }
        sum += sample;
      }
      mono[i] = sum / channels;
    }

    return {
      sampleRate,
      samples: mono,
    };
  }

  readFourCC(view, offset) {
    return String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3)
    );
  }

  linearResample(input, srcRate, dstRate) {
    if (srcRate === dstRate) {
      return input;
    }
    if (input.length === 0) {
      return new Float32Array(0);
    }
    const outLength = Math.max(1, Math.round((input.length * dstRate) / srcRate));
    const out = new Float32Array(outLength);
    if (outLength === 1) {
      out[0] = input[0];
      return out;
    }
    const scale = (input.length - 1) / (outLength - 1);
    for (let i = 0; i < outLength; i += 1) {
      const x = i * scale;
      const i0 = Math.floor(x);
      const i1 = Math.min(i0 + 1, input.length - 1);
      const t = x - i0;
      out[i] = input[i0] * (1 - t) + input[i1] * t;
    }
    return out;
  }

  toPcm16FloatScale(floatSamples) {
    const out = new Float32Array(floatSamples.length);
    for (let i = 0; i < floatSamples.length; i += 1) {
      const s = Math.max(-1, Math.min(1, floatSamples[i]));
      const i16 = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
      out[i] = i16;
    }
    return out;
  }

  async dispose() {
    await this.stop();
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    if (this.decodeAudioContext) {
      try {
        await this.decodeAudioContext.close();
      } catch (_error) {
        // Ignore decode context close errors during dispose.
      }
      this.decodeAudioContext = null;
    }
    this.initialized = false;
  }
}
