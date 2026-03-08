class DexterWakewordProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 1280;
    this.buffer = new Float32Array(this.bufferSize);
    this.pos = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) {
      return true;
    }

    for (let i = 0; i < input.length; i += 1) {
      this.buffer[this.pos] = input[i];
      this.pos += 1;
      if (this.pos === this.bufferSize) {
        this.port.postMessage(this.buffer.slice());
        this.pos = 0;
      }
    }

    return true;
  }
}

registerProcessor('dexter-wakeword-processor', DexterWakewordProcessor);
