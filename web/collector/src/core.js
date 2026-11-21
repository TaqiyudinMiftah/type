import { collectionConfig, validCollectionConfig } from './quota.js';
export const APP_VERSION = '2.3.4';
export const SCHEMA = 'ketik-web-session/v2';
export const LEGACY_SCHEMA = 'ketik-web-session/v1';
export const ROLES = ['calibration', 'test1', 'test2'];
export const LETTERS = 'abcdefghijklmnopqrstuvwxyz';
export const ROLE_NAMES = { calibration: 'Kalibrasi', test1: 'Uji pertama', test2: 'Uji kedua' };
export const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function localDay(date, timezone) {
  const p = Object.fromEntries(new Intl.DateTimeFormat('en-CA', { timeZone: timezone, year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(new Date(date)).map(x => [x.type, x.value]));
  return `${p.year}-${p.month}-${p.day}`;
}
export function nextRole(study, today = localDay(Date.now(), study.timezone)) {
  for (let i = 0; i < ROLES.length; i++) {
    const role = ROLES[i], day = study.days[role];
    if (!day) return i && study.days[ROLES[i - 1]] >= today ? ROLES[i - 1] : role;
    if (day > today) throw Error('Tanggal perangkat lebih awal dari sesi terakhir. Periksa jam perangkat.');
    if (day === today) return role;
  }
  return null;
}
export function newStudy(setup, previousParticipant) {
  return { schema: 'ketik-web-study/v2', collection: collectionConfig(), participant_id: previousParticipant || crypto.randomUUID(), study_id: crypto.randomUUID(), created_at: new Date().toISOString(), timezone: Intl.DateTimeFormat().resolvedOptions().timeZone, setup, days: { calibration: null, test1: null, test2: null } };
}
export function validateStudy(s) {
  if (!['ketik-web-study/v1', 'ketik-web-study/v2'].includes(s?.schema) || !UUID.test(s.participant_id) || !UUID.test(s.study_id) || !s.days || !s.setup) throw Error('Profil studi tidak valid.');
  if (s.schema === 'ketik-web-study/v2' && !validCollectionConfig(s.collection)) throw Error('Protokol kuota studi tidak valid.');
  new Intl.DateTimeFormat('en', { timeZone: s.timezone }).format();
  let previous = null;
  for (const r of ROLES) {
    const d = s.days[r];
    if (d !== null && (typeof d !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(d) || !Number.isFinite(Date.parse(d)) || new Date(d).toISOString().slice(0, 10) !== d)) throw Error('Tanggal studi tidak valid.');
    if (d && previous && d <= previous) throw Error('Hari studi harus berurutan dan berbeda.');
    if (d && ROLES.indexOf(r) && !previous) throw Error('Tahap studi sebelumnya belum ada.');
    previous = d;
  }
  if (['laptop', 'keyboard'].some(k => typeof s.setup[k] !== 'string' || !s.setup[k].trim() || s.setup[k].length > 120)) throw Error('Isi alias laptop dan keyboard.');
  return s;
}
export function countsFor(events) {
  const counts = Object.fromEntries([...LETTERS].map(c => [c, 0]));
  for (const e of events) if (e.type === 'keydown' && e.phase === 'typing' && !e.repeat && !e.is_composing && !e.ctrl && !e.alt && !e.meta) {
    const key = e.key?.toLowerCase();
    if (key?.length === 1 && key in counts) counts[key]++;
  }
  return counts;
}
export function interpolate(points, value) {
  if (points.length < 2 || value < points[0][0] || value > points.at(-1)[0]) return null;
  let left = 0, right = points.length - 1;
  while (right - left > 1) { const mid = (left + right) >> 1; if (points[mid][0] <= value) left = mid; else right = mid; }
  const [x, y] = points[left], [xx, yy] = points[right];
  return y + (yy - y) * (value - x) / (xx - x);
}
export function clockPoints(anchors, rate, firstFrame) {
  return anchors.filter(a => Number.isFinite(a.before_ms) && Number.isFinite(a.after_ms) && Number.isFinite(a.context_s) && a.after_ms >= a.before_ms && a.after_ms - a.before_ms <= 5)
    .map(a => [(a.before_ms + a.after_ms) / 2, a.context_s * rate - firstFrame])
    .filter((p, i, all) => !i || p[0] > all[i - 1][0]);
}
export function fitAffine(pairs) {
  if (pairs.length < 2) return null;
  const x = pairs.reduce((s, p) => s + p[0], 0) / pairs.length;
  const y = pairs.reduce((s, p) => s + p[1], 0) / pairs.length;
  const den = pairs.reduce((s, p) => s + (p[0] - x) ** 2, 0);
  if (!den) return null;
  const slope = pairs.reduce((s, p) => s + (p[0] - x) * (p[1] - y), 0) / den;
  return { slope, intercept: y - slope * x };
}
export function analyzeTiming(pcm, rate, firstFrame, events, anchors, captureIssues = []) {
  const points = clockPoints(anchors, rate, firstFrame), n = Math.round(rate * .005), energy = [];
  for (let i = 0; i + n <= pcm.length; i += n) {
    let v = 0; for (let j = i; j < i + n; j++) v += pcm[j] ** 2;
    energy.push(Math.sqrt(v / n));
  }
  const quiet = energy.slice(Math.round(.5 * rate / n), Math.round(2.5 * rate / n));
  const noise = Math.sqrt(quiet.reduce((s, e) => s + e * e, 0) / Math.max(1, quiet.length));
  const calibration = events.filter(e => e.type === 'keydown' && e.calibration === true);
  const markers = calibration.map(e => {
    const frame = interpolate(points, e.time_ms);
    if (frame === null) return { event_seq: e.seq, phase: e.phase, reason: 'outside_clock' };
    const start = Math.max(0, Math.ceil((frame - .25 * rate) / n)), end = Math.min(energy.length, Math.floor((frame + .25 * rate) / n));
    let best = start; for (let i = start; i < end; i++) if (energy[i] > energy[best]) best = i;
    if (end <= start || energy[best] <= Math.max(1e-4, 4 * noise)) return { event_seq: e.seq, phase: e.phase, reason: 'weak_signal' };
    const competing = energy.some((v, i) => i >= start && i < end && Math.abs(i - best) * n > .025 * rate && v >= .9 * energy[best]);
    return { event_seq: e.seq, phase: e.phase, clock_frame: frame, peak_frame: best * n + n / 2, reason: competing ? 'ambiguous_peak' : null };
  });
  const train = markers.filter((m, i) => i % 2 === 0 && !m.reason), validation = markers.filter((m, i) => i % 2 === 1 && !m.reason);
  const affine = fitAffine(train.map(m => [m.clock_frame, m.peak_frame]));
  const residuals = affine ? validation.map(m => (m.peak_frame - (affine.intercept + affine.slope * m.clock_frame)) / rate * 1000) : [];
  const issues = [...captureIssues];
  if (markers.length !== 16 || ['sync_start', 'sync_end'].some(p => markers.filter(m => m.phase === p).length !== 8)) issues.push('incomplete_calibration');
  if (markers.some(m => m.reason) || train.length !== 8 || validation.length !== 8) issues.push('uncertain_calibration');
  if (!affine || Math.abs(affine.slope - 1) > .005 || residuals.some(v => Math.abs(v) > 20)) issues.push('timing_residual');
  if (points.length < 2 || points.some((p, i) => i && p[1] < points[i - 1][1])) issues.push('non_monotonic_clock');
  if (events.some(e => typeof e.time_ms === 'number' && (e.handler_ms - e.time_ms > 20 || e.handler_ms < e.time_ms - 1))) issues.push('event_dispatch_lag');
  return { method: 'calibration-affine-energy-v1', status: issues.length ? 'needs_review' : 'checked', first_frame: firstFrame, sample_rate: rate, anchors, calibration: markers, affine, residuals_ms: residuals, quiet_rms: noise, issues: [...new Set(issues)], interpretation: 'Konsistensi penanda terhadap transien audio; bukan sertifikasi onset mekanik.' };
}
export function wavBytes(pcm, rate) {
  const data = new Uint8Array(44 + pcm.length * 4), view = new DataView(data.buffer);
  const str = (at, s) => { for (let i = 0; i < s.length; i++) data[at + i] = s.charCodeAt(i); };
  str(0, 'RIFF'); view.setUint32(4, 36 + pcm.length * 4, true); str(8, 'WAVE'); str(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 3, true); view.setUint16(22, 1, true); view.setUint32(24, rate, true);
  view.setUint32(28, rate * 4, true); view.setUint16(32, 4, true); view.setUint16(34, 32, true); str(36, 'data'); view.setUint32(40, pcm.length * 4, true);
  for (let i = 0; i < pcm.length; i++) view.setFloat32(44 + i * 4, pcm[i], true);
  return data;
}
export async function hash(bytes) { return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(v => v.toString(16).padStart(2, '0')).join(''); }

