export class WakewordAudioCaptureController {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.sourceNode = null;
    this.workletNode = null;
    this.sinkNode = null;
  }

  async start(onFrame) {
    if (this.audioContext) {
      return;
    }

    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    this.audioContext = new AudioContext({ sampleRate: 16000 });
    const workletUrl = new URL('./audio-worklet-processor.js', import.meta.url);
    await this.audioContext.audioWorklet.addModule(workletUrl);

    this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.workletNode = new AudioWorkletNode(this.audioContext, 'dexter-wakeword-processor');
    this.sinkNode = this.audioContext.createGain();
    this.sinkNode.gain.value = 0;

    this.workletNode.port.onmessage = (event) => {
      const chunk = event.data;
      if (!(chunk instanceof Float32Array)) {
        return;
      }
      onFrame(chunk);
    };

    this.sourceNode.connect(this.workletNode);
    this.workletNode.connect(this.sinkNode);
    this.sinkNode.connect(this.audioContext.destination);
  }

  async stop() {
    if (this.workletNode) {
      this.workletNode.port.onmessage = null;
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    if (this.sinkNode) {
      this.sinkNode.disconnect();
      this.sinkNode = null;
    }

    if (this.mediaStream) {
      for (const track of this.mediaStream.getTracks()) {
        track.stop();
      }
      this.mediaStream = null;
    }

    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }
  }
}
