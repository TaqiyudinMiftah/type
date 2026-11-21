import { test } from 'node:test';
import assert from 'node:assert/strict';
import { analyzeTiming, countsFor, localDay, newStudy, nextRole, validateStudy, wavBytes } from '../src/core.js';
import { BANK, choosePrompt } from '../src/prompts.js';
import { fixture } from './fixture.mjs';

test('three-day study keeps same-day additions and supports timezone boundaries', () => {
  const s = newStudy({ laptop: 'laptop', keyboard: 'internal' });
  s.timezone = 'Asia/Jakarta';
  assert.equal(localDay('2026-11-20T18:00:00Z', s.timezone), '2026-11-21');
  assert.equal(nextRole(s, '2026-11-20'), 'calibration');
  s.days.calibration = '2026-11-20';
  assert.equal(nextRole(s, '2026-11-20'), 'calibration');
  assert.equal(nextRole(s, '2026-11-21'), 'test1');
  s.days.test1 = '2026-11-21'; s.days.test2 = '2026-11-22';
  assert.equal(nextRole(s, '2026-11-23'), null);
  assert.throws(() => nextRole(s, '2026-11-19'));
  s.days.test2 = '2026-11-21'; assert.throws(() => validateStudy(s));
});
test('labels come from actual keys; repeat, shortcuts and composition are not counted', () => {
  const { events } = fixture();
  const a = events.find(e => e.key === 'a');
  const count = countsFor([...events, { ...a, repeat: true }, { ...a, ctrl: true }, { ...a, is_composing: true }]);
  assert.equal(count.a, 1); assert.equal(count.b, 1); assert.equal(count.q, 0);
});
test('independent acoustic pulses recover the input offset with held-out validation', () => {
  const f = fixture(), r = analyzeTiming(f.pcm, f.rate, f.first, f.events, f.anchors);
  assert.equal(r.status, 'checked');
  assert.ok(Math.abs(r.affine.slope - 1) < .0001);
  assert.ok(Math.abs(r.affine.intercept / f.rate * 1000 - 42.5) < 3);
  assert.equal(r.residuals_ms.length, 8); assert.ok(r.residuals_ms.every(n => Math.abs(n) < 6));
});
test('missing calibration, competing peaks and delayed dispatch remain unverified', () => {
  const f = fixture({ secondPeak: true });
  assert.equal(analyzeTiming(f.pcm, f.rate, f.first, f.events, f.anchors).status, 'needs_review');
  const g = fixture(); g.events.pop(); g.events[0].handler_ms += 30;
  const result = analyzeTiming(g.pcm, g.rate, g.first, g.events, g.anchors);
  assert.ok(result.issues.includes('incomplete_calibration')); assert.ok(result.issues.includes('event_dispatch_lag'));
});
test('WAV exports float32 samples without resampling', () => {
  const pcm = new Float32Array([0, -.125, .5]), view = new DataView(wavBytes(pcm, 48000).buffer);
  assert.equal(view.getUint16(20, true), 3); assert.equal(view.getUint32(24, true), 48000);
  assert.equal(view.getFloat32(48, true), -.125); assert.equal(view.getUint32(40, true), 12);
});
test('prompt banks are disjoint by role and every role covers a-z', () => {
  const all = Object.values(BANK).flat(); assert.equal(new Set(all).size, all.length);
  for (const bank of Object.values(BANK)) for (const c of 'abcdefghijklmnopqrstuvwxyz') assert.ok(bank.join(' ').includes(c), c);
  const counts = Object.fromEntries([...'abcdefghijklmnopqrstuvwxyz'].map(c => [c, 100])); counts.x = 0;
  assert.ok(choosePrompt('calibration', counts, []).text.includes('x'));
});
