import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CLASSES, PER_CLASS, TOTAL_TARGET, emptyCounts, assignQuota, selectionReport, eventClass } from '../src/quota.js';
import { SENTENCES, sentencePrompt, topupPrompt } from '../src/quota-prompts.js';
import { SignalEnvelope } from '../src/signal.js';
import { fixture } from './fixture.mjs';
import { additionalCounts } from '../src/extra-keys.js';
import { sessionTitle, sessionFilename } from '../src/session-label.js';
import { analyzeTiming } from '../src/core.js';

const key = (name, patch = {}) => ({ type: 'keydown', phase: 'typing', key: name === 'space' ? ' ' : name, code: name === 'space' ? 'Space' : `Key${name.toUpperCase()}`, repeat: false, is_composing: false, ctrl: false, alt: false, meta: false, ...patch });
test('exactly 25 first presses per class, including Space, with all surplus retained', () => {
  const events = [], selected = emptyCounts();
  const add = row => { row.seq = events.length; if (row.type === 'keydown') assignQuota(row, selected); events.push(row); };
  for (const c of CLASSES) for (let i = 0; i < 30; i++) add(key(c));
  add(key('space', { phase: 'sync_start', calibration: true })); add(key('a', { repeat: true })); add(key('b', { ctrl: true })); add(key('c', { is_composing: true }));
  add(key('a', { type: 'keyup' }));
  const result = selectionReport(events, true);
  assert.equal(TOTAL_TARGET, 675); assert.equal(result.selected_total, 675); assert.equal(result.complete, true);
  for (const c of CLASSES) { assert.equal(result.selected_counts[c], 25); assert.equal(result.raw_counts[c], 30); }
  assert.equal(result.selected_sequences.includes(25), false);
  events[25].quota_selected = true; assert.throws(() => selectionReport(events, true), /kuota/);
});
test('labels reflect actual keys and support uppercase without counting shortcuts', () => {
  assert.equal(eventClass(key('a', { key: 'Z', shift: true })), 'z');
  assert.equal(eventClass(key('space', { alt: true })), null);
  assert.equal(eventClass(key('space', { code: 'KeyA' })), null);
  assert.equal(eventClass(key('a', { phase: 'sync_end' })), null);
});
test('each stage has a distinct sentence containing every class below its quota', () => {
  assert.equal(new Set(Object.values(SENTENCES)).size, 3);
  for (const [role, text] of Object.entries(SENTENCES)) {
    assert.equal(sentencePrompt(role).kind, 'sentence');
    for (const c of CLASSES) { const count = [...text.toLowerCase()].filter(k => k === (c === 'space' ? ' ' : c)).length; assert.ok(count > 0 && count < PER_CLASS, `${role}:${c}`); }
  }
});
test('topups are deterministic, prioritize deficits and never request a full class', () => {
  const counts = Object.fromEntries(CLASSES.map(c => [c, 25])); counts.q = 0; counts.v = 23;
  const a = topupPrompt('calibration', counts, 'study', 1), b = topupPrompt('calibration', counts, 'study', 1);
  assert.deepEqual(a, b);
  assert.equal(a.keys.length, 27);
  assert.deepEqual(a.keys.slice(0, 8), Array(8).fill('q'));
  assert.equal(a.keys.filter(key => key === 'q').length, 25);
  assert.equal(a.keys.filter(key => key === 'v').length, 2);
  counts.q = 25; counts.v = 25; counts.space = 23;
  const end = topupPrompt('test1', counts, 'study', 2); assert.deepEqual(end.keys, ['space', 'space']); assert.equal(end.text, '  ');
  counts.space = 25; assert.equal(topupPrompt('test1', counts, 'study', 3).keys.length, 0);
});
function check(f) { const envelope = new SignalEnvelope(f.rate); for (let i = 0; i < f.pcm.length; i += 1007) envelope.add({ first_frame: f.first + i, data: f.pcm.subarray(i, i + 1007) }); return envelope.assess(f.events.filter(e => e.phase === 'sync_start'), f.anchors); }
test('real 5 ms envelopes survive arbitrary chunk boundaries and qualify eight clear taps', () => {
  const result = check(fixture()); assert.equal(result.status, 'passed'); assert.equal(result.passed, 8);
  assert.equal(result.quiet_rms, 0); assert.ok(result.markers.every(m => m.peak_rms > m.threshold));
});
test('inaudible and clipped taps block readiness; audible competing peaks only flag timing', () => {
  const weak = fixture(); weak.pcm.fill(1e-5); const rejected = check(weak); assert.equal(rejected.status, 'needs_review'); assert.equal(rejected.passed, 0);
  const echo = check(fixture({ secondPeak: true })); assert.equal(echo.status, 'passed'); assert.ok(echo.markers.every(m => m.timing_ambiguous));
  const clipped = fixture(); for (let i = 0; i < clipped.pcm.length; i++) if (clipped.pcm[i]) clipped.pcm[i] = 1.2;
  assert.equal(check(clipped).markers[0].reason, 'clipping');
});
test('noise-relative readiness is stable across gain, DC offset, and a brief quiet-period outlier', () => {
  const results = [];
  for (const gain of [1, .1, .01]) {
    const f = fixture();
    for (let i = 0; i < f.pcm.length; i++) f.pcm[i] = .02 + gain * (f.pcm[i] * .1 + Math.SQRT2 * .005 * Math.sin(2 * Math.PI * 1000 * i / f.rate));
    // One accidental transient while measuring the reference must not dominate its RMS.
    for (let i = f.rate; i < f.rate + 80; i++) f.pcm[i] += gain * .2;
    const r = check(f); results.push(r);
    assert.equal(r.method, 'key-signal-v3'); assert.equal(r.status, 'passed');
    assert.ok(r.markers.every(m => m.snr_db > 6 && m.snr_db < 12));
    assert.ok(Math.abs(r.quiet_rms - gain * .005) < 1e-7);
  }
  assert.ok(Math.abs(results[0].markers[0].snr_db - results[2].markers[0].snr_db) < .01);
  const noise = fixture(); for (let i = 0; i < noise.pcm.length; i++) noise.pcm[i] = .005 * Math.SQRT2 * Math.sin(2 * Math.PI * 1000 * i / noise.rate);
  assert.equal(check(noise).passed, 0);
});
test('audibility survives unusable performance anchors while the strict timing audit still fails', () => {
  const f = fixture();
  for (const e of f.events) e.audio_context_s = f.first / f.rate + (e.time_ms - 1000) / 1000;
  for (const anchor of f.anchors) anchor.after_ms = anchor.before_ms + 12;
  const report = check(f);
  assert.equal(report.status,'passed'); assert.equal(report.diagnostics.usable_performance_anchors,0);
  assert.equal(report.diagnostics.audio_context_markers,8);
  assert.ok(report.markers.every(m => m.frame_source === 'audio-context-snapshot'));
  assert.equal(analyzeTiming(f.pcm,f.rate,f.first,f.events,f.anchors).status,'needs_review');
  for (const e of f.events) e.time_ms += 1_700_000_000_000;
  assert.equal(check(f).status,'passed');
  f.pcm.fill(1e-5); assert.equal(check(f).passed,0);
  for (const e of f.events) delete e.audio_context_s;
  assert.equal(check(f).markers[0].reason,'outside_clock');
});
test('physical extra keys retain modifier sides without inflating the a-z and Space quota', () => {
  const events = [key('Control', { code:'ControlLeft',ctrl:true }), key('Control', {code:'ControlRight',ctrl:true}), key('Backspace',{code:'Backspace'}), key('Tab',{code:'Tab'}), key('Shift',{code:'ShiftLeft',shift:true}), key('Control',{code:'ControlLeft',ctrl:true,repeat:true})];
  for (const e of events) e.quota_selected = false;
  const counts = additionalCounts(events);
  assert.equal(counts.ControlLeft,1); assert.equal(counts.ControlRight,1); assert.equal(counts.Backspace,1); assert.equal(counts.ShiftLeft,1);
  assert.equal(selectionReport(events,true).selected_total,0);
});
test('session names use the reported device and legacy names survive without editing sealed data', () => {
  const row={study:{setup:{laptop:'Legacy Acer'}},local_day:'2026-11-20',role:'calibration',session_id:'abcdefgh-1234'};
  assert.equal(sessionTitle(row),'Legacy Acer');
  row.study.setup.device_name='Lenovo / ThinkPad T14';
  assert.equal(sessionTitle(row),'Lenovo / ThinkPad T14');
  assert.equal(sessionFilename(row),'ketik-Lenovo-ThinkPad-T14-2026-11-20-calibration-abcdefgh.zip');
});
