import { CLASSES, displayKey } from './quota.js';
import { EXTRA_KEYS } from './extra-keys.js';

export class AudioFeedback {
  constructor(canvas, level, status, meter) {
    this.canvas = canvas;
    this.level = level;
    this.status = status;
    this.meter = meter;
    this.relative = document.getElementById('relative-level');
    this.baseline = document.getElementById('noise-floor');
    this.history = new Array(32).fill(0);
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  }
  connect(capture) { this.disconnect(); this.capture = capture; this.data = new Float32Array(capture.analyzer.fftSize); this.frame = requestAnimationFrame(t => this.draw(t)); }
  setNoiseReference(profile) {
    this.reference = profile;
    if (this.baseline) this.baseline.textContent = profile ? `Latar perangkat: ${Math.round(20 * Math.log10(Math.max(profile.quiet_raw_rms, 1e-6)))} dBFS · menjadi referensi 0 dB.` : 'Referensi latar diukur saat tiga detik hening.';
  }
  draw(time) {
    if (!this.capture) return;
    if (!this.last || time - this.last >= (this.reduced ? 200 : 33)) {
      this.last = time;
      this.capture.analyzer.getFloatTimeDomainData(this.data);
      let squares = 0, sum = 0, peak = 0;
      for (const x of this.data) { squares += x*x; sum += x; peak = Math.max(peak, Math.abs(x)); }
      const rms = Math.sqrt(squares / this.data.length), db = 20 * Math.log10(Math.max(rms, 1e-6));
      const ac = Math.sqrt(Math.max(0, squares / this.data.length - (sum / this.data.length) ** 2));
      const above = this.reference ? Math.max(0, 20 * Math.log10(Math.max(ac, 1e-6) / Math.max(this.reference.quiet_rms, 1e-6))) : null;
      this.level.textContent = `${Math.round(db)} dBFS`;
      this.meter.value = db;
      const strength = this.reference ? Math.min(1, above / 24) : Math.min(1, Math.max(0, (db + 90) / 80));
      if (this.relative) this.relative.textContent = above === null ? 'Mengukur referensi…' : `+${Math.round(above)} dB di atas latar`;
      this.history.shift(); this.history.push(strength);
      const canvas = this.canvas, width = canvas.clientWidth || 400, height = canvas.clientHeight || 60, scale = Math.min(2, devicePixelRatio || 1);
      if (canvas.width !== Math.round(width * scale)) canvas.width = Math.round(width * scale);
      if (canvas.height !== Math.round(height * scale)) canvas.height = Math.round(height * scale);
      const ctx = canvas.getContext('2d'); ctx.setTransform(scale, 0, 0, scale, 0, 0); ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = peak >= .999 ? '#f2a4a0' : '#60c985';
      const step = width / 32;
      for (let i = 0; i < 32; i++) { const h = 3 + this.history[i] * (height - 8); ctx.beginPath(); ctx.roundRect(i * step + 2, (height-h)/2, Math.max(2, step-5), h, 4); ctx.fill(); }
      canvas.dataset.active = 'true'; canvas.dataset.level = String(Math.round(db));
      if (peak >= .999) this.setStatus('Clipping · turunkan gain mikrofon', 'error');
    }
    this.frame = requestAnimationFrame(t => this.draw(t));
  }
  setStatus(text, kind = '') { if (this.status.textContent !== text) this.status.textContent = text; this.status.dataset.kind = kind; }
  disconnect() {
    cancelAnimationFrame(this.frame);
    this.capture = null;
    this.history.fill(0);
    this.last = null;
    this.level.textContent = '— dBFS';
    this.meter.value = this.meter.min;
    this.relative.textContent = 'Mikrofon tidak aktif';
    this.canvas.dataset.active = 'false';
    this.canvas.getContext('2d').clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}
export function createExtraKeyboard(container) {
  for (const [key, label] of Object.entries(EXTRA_KEYS)) {
    const cap = document.createElement('div'); cap.className = 'extra-keycap'; cap.dataset.extraKey = key;
    const name = document.createElement('span'); name.textContent = label;
    const count = document.createElement('small'); count.textContent = '0 ketukan'; cap.append(name, count); container.append(cap);
  }
  return {
    update(counts) { for (const [key, n] of Object.entries(counts)) container.querySelector(`[data-extra-key="${key}"] small`).textContent = `${n} ketukan`; },
    press(key, down) { container.querySelector(`[data-extra-key="${key}"]`)?.classList.toggle('pressed', down); },
    clear() { for (const cap of container.querySelectorAll('.pressed')) cap.classList.remove('pressed'); }
  };
}
export function createKeyboard(container) {
  for (const keys of ['qwertyuiop', 'asdfghjkl', 'zxcvbnm', ['space']]) {
    const row = document.createElement('div'); row.className = 'keyboard-row';
    for (const key of keys) {
      const cap = document.createElement('div'); cap.className = `keycap${key === 'space' ? ' space-key' : ''}`; cap.dataset.key = key;
      const name = document.createElement('span'); name.textContent = displayKey(key);
      const count = document.createElement('small'); count.textContent = '0/25'; cap.append(name, count); row.append(cap);
    }
    container.append(row);
  }
  return {
    update(counts, expected) { for (const key of CLASSES) { const cap = container.querySelector(`[data-key="${key}"]`); cap.querySelector('small').textContent = `${counts[key]}/25`; cap.classList.toggle('enough', counts[key] === 25); cap.classList.toggle('expected', expected === key); } },
    press(key, down) { container.querySelector(`[data-key="${key}"]`)?.classList.toggle('pressed', down); },
    clear() { for (const cap of container.querySelectorAll('.pressed')) cap.classList.remove('pressed'); }
  };
}
