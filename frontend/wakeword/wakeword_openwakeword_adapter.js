import { WakewordAudioCaptureController } from './audio_capture_controller.js';

const DEFAULT_BACKEND_ORIGIN =
  window.location.port === '5173'
    ? `${window.location.protocol}//${window.location.hostname}:5050`
    : window.location.origin;

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

    this.worker.postMessage({
      type: 'init',
      payload: {
        melspecModelUrl: `${backendOrigin}/models/openwakeword_resources/melspectrogram.onnx`,
        embeddingModelUrl: `${backendOrigin}/models/openwakeword_resources/embedding_model.onnx`,
        wakewordModelUrls: {
          dexter_start: `${backendOrigin}/models/wakewords/dexter_start.onnx`,
          dexter_stop: `${backendOrigin}/models/wakewords/dexter_stop.onnx`,
          dexter_abort: `${backendOrigin}/models/wakewords/dexter_abort.onnx`,
        },
        thresholds: {
          dexter_start: 0.56,
          dexter_stop: 0.66,
          dexter_abort: 0.66,
        },
        cooldownMs: {
          dexter_start: 1000,
          dexter_stop: 1000,
          dexter_abort: 1000,
        },
        patienceFrames: {
          dexter_start: 2,
          dexter_stop: 2,
          dexter_abort: 2,
        },
        warmupMs: 1500,
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
    for (const file of files) {
      const arrayBuffer = await file.arrayBuffer();
      const decoded = await this.decodeFileTo16kMonoFloat32(file.name, arrayBuffer);
      prepared.push({
        name: file.name,
        sampleRateInput: decoded.sampleRateInput,
        sampleRateRuntime: decoded.sampleRateRuntime,
        samples: decoded.samples,
      });
      transferList.push(decoded.samples.buffer);
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
