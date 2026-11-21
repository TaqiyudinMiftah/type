class Recorder extends AudioWorkletProcessor {
  constructor() {
    super(); this.active = true; this.pending = []; this.count = 0; this.first = null; this.nextFrame = null; this.emitted = false; this.warmupDiscarded = 0;
    this.port.onmessage = e => { if (e.data === 'stop') { this.flush(); this.active = false; this.port.postMessage({ type: 'stopped' }); } };
  }
  flush() {
    if (!this.count) return;
    const data = new Float32Array(this.count); let at = 0;
    for (const block of this.pending) { data.set(block, at); at += block.length; }
    this.port.postMessage({ type: 'audio', first_frame: this.first, frames: this.count, warmup_discarded_frames: this.warmupDiscarded, data }, [data.buffer]);
    this.emitted = true;
    this.pending = []; this.count = 0; this.first = null;
  }
  process(inputs, outputs) {
    for (const out of outputs) for (const channel of out) channel.fill(0);
    if (!this.active) return false;
    const channels = inputs[0];
    if (!channels?.length || !channels[0].length) {
      if (this.emitted) this.port.postMessage({ type: 'gap', reason: 'input_unavailable', at: currentFrame });
      else { this.warmupDiscarded += this.count; this.pending = []; this.count = 0; this.first = null; this.nextFrame = null; }
      return true;
    }
    const data = new Float32Array(channels[0].length);
    for (const channel of channels) for (let i = 0; i < data.length; i++) data[i] += channel[i] / channels.length;
    if (this.nextFrame !== null && this.nextFrame !== currentFrame) {
      const expected = this.nextFrame;
      if (this.emitted) { this.flush(); this.port.postMessage({ type: 'gap', reason: 'frame_discontinuity', at: currentFrame, expected }); }
      else { this.warmupDiscarded += this.count; this.pending = []; this.count = 0; }
      this.first = currentFrame;
    }
    if (this.first === null) this.first = currentFrame;
    this.nextFrame = currentFrame + data.length;
    this.pending.push(data); this.count += data.length;
    if (this.count >= sampleRate) this.flush();
    return true;
  }
}
registerProcessor('ketik-recorder', Recorder);
