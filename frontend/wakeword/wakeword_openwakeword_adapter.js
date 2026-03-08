import { WakewordAudioCaptureController } from './audio_capture_controller.js';

const DEFAULT_BACKEND_ORIGIN = `${window.location.protocol}//${window.location.hostname}:5000`;

export class OpenWakeWordAdapter {
  constructor(config = {}) {
    this.config = config;
    this.listeners = new Map();
    this.audio = new WakewordAudioCaptureController();
    this.worker = null;
    this.initialized = false;
    this.running = false;
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

    this.worker = new Worker(new URL('./inference_worker.js', import.meta.url));
    this.worker.onmessage = (event) => {
      const { type, payload } = event.data || {};
      if (!type) {
        return;
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
          dexter_start: 0.55,
          dexter_stop: 0.6,
          dexter_abort: 0.6,
        },
        cooldownMs: {
          dexter_start: 1000,
          dexter_stop: 1000,
          dexter_abort: 1000,
        },
      },
    });

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

  async dispose() {
    await this.stop();
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.initialized = false;
  }
}
