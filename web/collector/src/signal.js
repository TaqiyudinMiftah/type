import { clockPoints, interpolate } from './core.js';

export const SIGNAL_METHOD = 'key-signal-v3';
export const MIN_SNR_DB = 6;
export const SIGNAL_FLOOR = 1e-6;
const percentile = (sorted, p) => sorted[Math.floor((sorted.length - 1) * p)];

// Tiny envelopes stay on the main thread; the raw recording still goes to its worker.
export class SignalEnvelope {
  constructor(rate) { this.rate = rate; this.size = Math.round(rate * .005); this.energy = []; this.rawEnergy = []; this.peaks = []; this.count = 0; this.square = 0; this.sum = 0; this.peak = 0; this.frames = 0; }
  add(block) {
    this.firstFrame ??= block.first_frame;
    for (const value of block.data) {
      this.square += value * value; this.sum += value; this.peak = Math.max(this.peak, Math.abs(value)); this.count++; this.frames++;
      if (this.count === this.size) {
        const power = this.square / this.size, mean = this.sum / this.size;
        this.energy.push(Math.sqrt(Math.max(0, power - mean * mean)));
        this.rawEnergy.push(Math.sqrt(power)); this.peaks.push(this.peak);
        this.count = 0; this.square = 0; this.sum = 0; this.peak = 0;
      }
    }
  }
  noiseProfile() {
    const start = Math.round(.5 * this.rate / this.size), end = Math.round(2.5 * this.rate / this.size);
    if (this.energy.length < end) return null;
    const data = this.energy.slice(start, end).sort((a, b) => a - b);
    const raw = this.rawEnergy.slice(start, end).sort((a, b) => a - b);
    return { quiet_rms: percentile(data, .5), quiet_p95_rms: percentile(data, .95), quiet_raw_rms: percentile(raw, .5) };
  }
  quiet() { return this.noiseProfile()?.quiet_rms ?? null; }
  tap(event, anchors) {
    const profile = this.noiseProfile(), noise = profile?.quiet_rms;
    if (!profile) return { event_seq: event.seq, reason: 'baseline_incomplete' };
    // Readiness uses the same AudioContext clock as the PCM. Event timestamps and
    // strict performance-clock anchors remain available for the separate timing audit.
    const snapshot = Number.isFinite(event.audio_context_s) && event.audio_context_s >= 0;
    const frame = snapshot ? event.audio_context_s * this.rate - this.firstFrame : interpolate(clockPoints(anchors, this.rate, this.firstFrame), event.time_ms);
    const mapping = { frame_source: snapshot ? 'audio-context-snapshot' : 'performance-clock-interpolation', ...(snapshot ? { event_audio_context_s: event.audio_context_s } : {}) };
    if (frame === null) return { event_seq: event.seq, reason: 'outside_clock', ...mapping };
    const start = Math.max(0, Math.ceil((frame - .25 * this.rate) / this.size));
    const end = Math.min(this.energy.length, Math.floor((frame + .25 * this.rate) / this.size));
    if (end <= start) return { event_seq: event.seq, reason: 'outside_audio', clock_frame: frame, ...mapping };
    let best = start, clipped = false;
    for (let i = start; i < end; i++) { if (this.energy[i] > this.energy[best]) best = i; if (this.peaks[i] >= .999) clipped = true; }
    const threshold = Math.max(SIGNAL_FLOOR, noise * 10 ** (MIN_SNR_DB / 20), profile.quiet_p95_rms * 1.2), level = this.energy[best];
    const reason = clipped ? 'clipping' : level <= threshold ? 'weak_signal' : null;
    const ambiguous = !reason && this.energy.slice(start, end).some((v, offset) => Math.abs(start + offset - best) * this.size > .025 * this.rate && v >= .9 * level);
    return { event_seq: event.seq, reason, peak_rms: level, peak_abs: this.peaks[best], threshold, snr_db: 20 * Math.log10(Math.max(level, 1e-12) / Math.max(noise, 1e-9)), timing_ambiguous: Boolean(ambiguous), clock_frame: frame, peak_frame: (best + .5) * this.size, ...mapping };
  }
  assess(events, anchors) {
    const markers = events.filter(e => e.calibration === true).map(e => this.tap(e, anchors));
    const passed = markers.filter(m => m.reason === null).length;
    return { method: SIGNAL_METHOD, status: markers.length === 8 && passed === 8 ? 'passed' : 'needs_review', sample_rate: this.rate, first_frame: this.firstFrame, frames: this.frames, ...this.noiseProfile(), baseline_estimator: 'median-5ms-ac-rms', min_snr_db: MIN_SNR_DB, floor_rms: SIGNAL_FLOOR, passed, total: markers.length, markers, diagnostics: { usable_performance_anchors: clockPoints(anchors, this.rate, this.firstFrame).length, audio_context_markers: markers.filter(m => m.frame_source === 'audio-context-snapshot').length }, interpretation: 'Keterdengaran memakai posisi AudioContext yang sama dengan PCM; pencocokan jam performa dan timing tetap diaudit terpisah.' };
  }
}
