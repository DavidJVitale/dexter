function floatToPcm16(floatBuffer) {
  const pcm16 = new Int16Array(floatBuffer.length);
  for (let i = 0; i < floatBuffer.length; i += 1) {
    const s = Math.max(-1, Math.min(1, floatBuffer[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm16;
}

function bytesToBase64(bytes) {
  let bin = "";
  const len = bytes.length;
  for (let i = 0; i < len; i += 1) {
    bin += String.fromCharCode(bytes[i]);
  }
  return btoa(bin);
}

function computeRms(floatBuffer) {
  if (!floatBuffer.length) {
    return 0;
  }
  let sumSq = 0;
  for (let i = 0; i < floatBuffer.length; i += 1) {
    const v = floatBuffer[i];
    sumSq += v * v;
  }
  return Math.sqrt(sumSq / floatBuffer.length);
}

export class AudioCapture {
  constructor({ onChunk }) {
    this.onChunk = onChunk;
    this.isCapturing = false;
    this.audioContext = null;
    this.processor = null;
    this.source = null;
    this.mediaStream = null;
    this.seq = 0;
  }

  async initialize() {
    if (this.audioContext) {
      return;
    }

    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    this.source = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (event) => {
      if (!this.isCapturing) {
        return;
      }
      const input = event.inputBuffer.getChannelData(0);
      const pcm16 = floatToPcm16(input);
      const bytes = new Uint8Array(pcm16.buffer);
      const payload = {
        seq: this.seq,
        sample_rate: 16000,
        channels: 1,
        format: "pcm16",
        pcm_b64: bytesToBase64(bytes),
        rms: computeRms(input),
      };
      this.seq += 1;
      this.onChunk(payload);
    };

    this.source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);
  }

  startCapture() {
    this.seq = 0;
    this.isCapturing = true;
  }

  stopCapture() {
    this.isCapturing = false;
  }
}
