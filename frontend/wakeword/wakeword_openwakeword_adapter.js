import { WakewordAudioCaptureController } from './audio_capture_controller.js';

const DEFAULT_BACKEND_ORIGIN =
  window.location.port === '5173'
    ? `${window.location.protocol}//${window.location.hostname}:5050`
    : window.location.origin;

const DEXTER_MODEL_PATH = '/models/wakewords/dexter.onnx';

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
    const configuredDexterModel = this.config.wakewordModelUrl || DEXTER_MODEL_PATH;
    const dexterModelUrl = /^https?:\/\//i.test(String(configuredDexterModel || ''))
      ? String(configuredDexterModel)
      : `${backendOrigin}${configuredDexterModel}`;

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
        wakewordModelUrls: { dexter: dexterModelUrl },
        thresholds: { dexter: this.config.threshold ?? 0.6 },
        cooldownMs: { dexter: this.config.cooldownMs ?? 1000 },
        patienceFrames: { dexter: this.config.patienceFrames ?? 2 },
        warmupMs: 1500,
        startupRequireArming: this.config.startupRequireArming ?? true,
        startupArmCeiling: this.config.startupArmCeiling ?? 0.9,
        startupArmFrames: this.config.startupArmFrames ?? 2,
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

    if (this.worker) {
      this.worker.postMessage({ type: 'reset' });
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

  async dispose() {
    await this.stop();
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.initialized = false;
  }
}
