import { analyzeTiming, countsFor, LEGACY_SCHEMA as SCHEMA, wavBytes, hash } from '../src/core.js';
export function fixture({ jitter = 0, secondPeak = false } = {}) {
  const rate = 16000, first = 4096, pcm = new Float32Array(36 * rate), events = [], anchors = [];
  const event = (time, key, phase, calibration = false) => {
    const e = { seq: events.length, type: 'keydown', phase, time_ms: 1000 + time * 1000, handler_ms: 1000 + time * 1000 + .1, is_trusted: false, key, code: key === ' ' ? 'Space' : `Key${key.toUpperCase()}`, repeat: false, is_composing: false, shift: false, ctrl: false, alt: false, meta: false };
    if (calibration) e.calibration = true;
    events.push(e);
    let frame = Math.floor((time + .04 + (events.length % 2 ? jitter : 0)) * rate / 80) * 80 + 40;
    for (let j = -5; j <= 5; j++) pcm[frame + j] = .4;
    if (secondPeak) for (let j = -5; j <= 5; j++) pcm[frame + 1600 + j] = .4;
  };
  for (let i = 0; i < 8; i++) event(3.1 + i * .83, ' ', 'sync_start', true);
  event(12, 'a', 'typing'); event(13, 'b', 'typing');
  for (let i = 0; i < 8; i++) event(26 + i * .83, ' ', 'sync_end', true);
  for (let t = 0; t <= 36; t += .5) anchors.push({ before_ms: 1000 + t * 1000, after_ms: 1000 + t * 1000, context_s: first / rate + t });
  return { pcm, rate, first, events, anchors };
}
export async function fixturePackage() {
  const f = fixture(), day = '2026-11-20', setup = { laptop: 'synthetic fixture', keyboard: 'synthetic keys', microphone: 'synthetic microphone' };
  const study = { schema: 'ketik-web-study/v1', participant_id: '11111111-1111-4111-8111-111111111111', study_id: '22222222-2222-4222-8222-222222222222', created_at: `${day}T00:00:00Z`, timezone: 'Asia/Jakarta', setup, days: { calibration: day, test1: null, test2: null } };
  const timing = analyzeTiming(f.pcm, f.rate, f.first, f.events, f.anchors);
  const encode = x => new TextEncoder().encode(x);
  const payload = {
    'audio.wav': wavBytes(f.pcm, f.rate),
    'events.jsonl': encode(f.events.map(e => JSON.stringify(e)).join('\n') + '\n'),
    'timing.json': encode(JSON.stringify(timing, null, 2) + '\n')
  };
  const session = { schema: SCHEMA, app_version: '1.0.0', session_id: '33333333-3333-4333-8333-333333333333', participant_id: study.participant_id, study_id: study.study_id, study, role: 'calibration', local_day: day, timezone: study.timezone, started_at: `${day}T01:00:00Z`, finished_at: `${day}T01:00:36Z`, status: 'complete', synthetic: true, sample_rate: f.rate, frames: f.pcm.length, channels: 1, format: 'IEEE_FLOAT_32_LE', first_frame: f.first, time_origin_epoch_ms: Date.parse(`${day}T00:59:59Z`), requested_settings: {}, track_settings: {}, consent: { version: 'local-audio-key-events-v1', accepted_at: `${day}T00:59:58Z` }, prompts: [{ id: 'synthetic', text: 'aaaaaaaa', version: 'synthetic' }], recorded_counts: countsFor(f.events), event_count: f.events.length, audio_blocks: [{ seq: 0, first_frame: f.first, frames: f.pcm.length }], capture_issues: [], timing_status: timing.status, files: {} };
  for (const [name, data] of Object.entries(payload)) session.files[name] = await hash(data);
  payload['session.json'] = encode(JSON.stringify(session, null, 2) + '\n');
  return payload;
}
