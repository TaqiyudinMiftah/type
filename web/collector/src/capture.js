import { MAX_AUDIO_BYTES } from './quota.js';

export class Capture {
  constructor(stream, onBlock, onIssue, onMeter, options = {}) {
    this.stream = stream; this.onBlock = onBlock; this.onIssue = onIssue; this.onMeter = onMeter;
    this.anchors = []; this.frames = 0; this.clipped = 0; this.stopping = false;
    this.maxBytes = options.maxBytes ?? MAX_AUDIO_BYTES;
  }
  anchor() {
    if (!this.context) return;
    const before_ms = performance.now(), context_s = this.context.currentTime, after_ms = performance.now();
    const anchor = { before_ms, after_ms, context_s }; this.anchors.push(anchor);
    return anchor;
  }
  async start() {
    let resolveReady;
    const ready = new Promise(resolve => { resolveReady = resolve; });
    try { this.context = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' }); }
    catch { this.context = new AudioContext(); }
    await this.context.resume();
    if (this.context.sampleRate > 96000 || this.context.sampleRate < 8000) throw Error('Sample rate browser tidak didukung.');
    await this.context.audioWorklet.addModule('./audio-worklet.js');
    this.node = new AudioWorkletNode(this.context, 'ketik-recorder', { outputChannelCount: [1] });
    this.source = this.context.createMediaStreamSource(this.stream);
    this.analyzer = this.context.createAnalyser(); this.analyzer.fftSize = 2048;
    this.source.connect(this.analyzer);
    this.source.connect(this.node); this.node.connect(this.context.destination); // Worklet emits silence.
    this.anchor();
    this.node.port.onmessage = e => {
      const m = e.data;
      if (m.type === 'stopped') { this.resolveStop?.(); return; }
      if (m.type === 'gap') { if (!this.stopping) this.onIssue(`audio_gap:${m.reason}`); return; }
      if (m.type !== 'audio') return;
      const remaining = Math.floor(this.maxBytes / 4) - this.frames;
      if (remaining <= 0) { if (!this.limited) { this.limited = true; this.onIssue('audio_byte_limit'); } return; }
      const limited = m.frames > remaining;
      if (limited) { m.data = m.data.slice(0, remaining); m.frames = remaining; }
      this.warmupDiscarded = m.warmup_discarded_frames;
      this.frames += m.frames;
      let square = 0, peak = 0;
      for (const x of m.data) { square += x * x; peak = Math.max(peak, Math.abs(x)); if (Math.abs(x) >= .999) this.clipped++; }
      this.onMeter?.(Math.sqrt(square / m.frames), peak);
      this.onBlock(m);
      resolveReady();
      if (limited || this.frames * 4 >= this.maxBytes) { this.limited = true; this.onIssue('audio_byte_limit'); }
    };
    this.node.onprocessorerror = () => this.onIssue('audio_processor_error');
    this.context.onstatechange = () => { if (!this.stopping && this.context.state !== 'running') this.onIssue('audio_suspended'); };
    this.track = this.stream.getAudioTracks()[0];
    this.track.onended = () => { if (!this.stopping) this.onIssue('microphone_ended'); };
    this.track.onmute = () => { if (!this.stopping) this.onIssue('microphone_muted'); };
    this.timer = setInterval(() => this.anchor(), 500);
    let timeout;
    try { await Promise.race([ready, new Promise((_, reject) => { timeout = setTimeout(() => reject(Error('Mikrofon tidak menghasilkan blok audio.')), 10000); })]); }
    finally { clearTimeout(timeout); }
    return this.context.sampleRate;
  }
  async stop() {
    if (this.stopping) return;
    this.stopping = true; clearInterval(this.timer); this.anchor();
    if (this.node) {
      await new Promise(resolve => {
        const timeout = setTimeout(() => { this.onIssue('stop_ack_timeout'); resolve(); }, 1500);
        this.resolveStop = () => { clearTimeout(timeout); resolve(); };
        this.node.port.postMessage('stop');
      });
      this.node.disconnect(); this.source.disconnect();
    }
    this.stream.getTracks().forEach(t => t.stop());
    await this.context?.close();
  }
}
