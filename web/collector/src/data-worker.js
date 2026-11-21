import { zipSync, unzipSync } from './vendor/fflate.js';
import { SCHEMA, LEGACY_SCHEMA, analyzeTiming, countsFor, hash, wavBytes, validateStudy } from './core.js';
import { MAX_AUDIO_BYTES, selectionReport } from './quota.js';
import { additionalCounts } from './extra-keys.js';
import { transaction, getChunks, getSession } from './storage.js';

const encoder = new TextEncoder(), decoder = new TextDecoder();
let chain = Promise.resolve();
const volatile = new Map();
const json = obj => encoder.encode(JSON.stringify(obj, null, 2) + '\n');
const put = row => transaction(['sessions'], 'readwrite', tx => tx.objectStore('sessions').put(row));

async function payload(id, recovery = false) {
  const row = volatile.get(id) || await getSession(id);
  if (!row) throw Error('Sesi tidak ditemukan di browser ini.');
  const chunks = await getChunks(id);
  if (!chunks.length) throw Error('Belum ada blok audio tersimpan untuk sesi ini.');
  if (row.status === 'recording' && recovery) {
    row.status = 'interrupted'; row.finished_at = row.updated_at;
    row.capture_issues = [...row.capture_issues, 'recovered_after_reload'];
  }
  const frames = chunks.reduce((sum, c) => sum + c.data.length, 0);
  const isV2 = row.schema === SCHEMA;
  if (isV2 ? frames * 4 > MAX_AUDIO_BYTES : frames > row.sample_rate * 301) throw Error('Audio paket melampaui batas sesi.');
  const pcm = new Float32Array(frames); let pos = 0;
  for (const [index, c] of chunks.entries()) {
    if (c.seq !== index || c.first_frame !== chunks[0].first_frame + pos || c.frames !== c.data.length) throw Error('Urutan blok audio tersimpan tidak utuh.');
    pcm.set(c.data, pos); pos += c.data.length;
  }
  const timing = row.sealed_timing || analyzeTiming(pcm, row.sample_rate, chunks[0].first_frame, row.events, row.anchors, row.capture_issues);
  let peak = 0, clipped = 0;
  for (const v of pcm) { if (!Number.isFinite(v)) throw Error('Audio mengandung sampel tidak finite.'); peak = Math.max(peak, Math.abs(v)); if (Math.abs(v) >= .999) clipped++; }
  const files = {
    'audio.wav': wavBytes(pcm, row.sample_rate),
    'events.jsonl': encoder.encode(row.events.map(e => JSON.stringify(e)).join('\n') + (row.events.length ? '\n' : '')),
    'timing.json': json(timing)
  };
  const hashes = Object.fromEntries(await Promise.all(Object.entries(files).map(async ([k, bytes]) => [k, await hash(bytes)])));
  const collection = isV2 ? selectionReport(row.events, true) : null;
  const session = row.sealed_session || {
    schema: isV2 ? SCHEMA : LEGACY_SCHEMA, app_version: row.app_version || (isV2 ? '2.2.0' : '1.0.0'), session_id: row.session_id,
    participant_id: row.study.participant_id, study_id: row.study.study_id, study: row.study,
    role: row.role, local_day: row.local_day, started_at: row.started_at, finished_at: row.finished_at,
    status: row.status, synthetic: row.synthetic === true, timezone: row.study.timezone,
    time_origin_epoch_ms: row.time_origin_epoch_ms, sample_rate: row.sample_rate, frames, channels: 1,
    format: 'IEEE_FLOAT_32_LE', first_frame: chunks[0].first_frame,
    warmup_discarded_frames: row.warmup_discarded_frames || 0,
    audio_route: 'getUserMedia → Web Audio downmix mono → AudioWorklet PCM',
    requested_settings: row.requested_settings, track_settings: row.track_settings, consent: row.consent,
    prompts: row.prompts, recorded_counts: isV2 ? collection.raw_counts : countsFor(row.events), event_count: row.events.length,
    ...(isV2 ? { collection, preflight: row.preflight, additional_keys: { method: 'physical-code-keydown-v1', scope: 'typing', counts: additionalCounts(row.events) } } : {}),
    audio_blocks: chunks.map(c => ({ seq: c.seq, first_frame: c.first_frame, frames: c.frames })),
    capture_issues: row.capture_issues, timing_status: timing.status,
    quality: { peak_abs: peak, clipped_samples: clipped, clipped_fraction: clipped / Math.max(1, frames) },
    files: hashes, clock_origin: 'browser-reported; dates and physical setup are participant-reported'
  };
  if (Object.keys(hashes).some(k => session.files[k] !== hashes[k])) throw Error('Paket berbeda dari sesi yang sudah dibekukan.');
  row.status = session.status; row.sealed_session = session; row.sealed_timing = timing;
  volatile.set(id, row);
  try { await put(row); } catch { /* A full local store must not prevent rescuing a ZIP. */ }
  return { row, session, timing, files: { ...files, 'session.json': json(session) } };
}

async function handle(message) {
  const { type, request_id } = message;
  if (type === 'start') {
    const { row, chunks } = message;
    const first = chunks[0]?.first_frame;
    let frames = 0;
    for (const [index, chunk] of chunks.entries()) {
      if (chunk.seq !== index || chunk.first_frame !== first + frames || chunk.frames !== chunk.data.length || !chunk.frames) {
        throw Error('Audio awal belum utuh. Klik Mulai sesi untuk mencoba lagi.');
      }
      frames += chunk.frames;
    }
    if (row.preflight.status !== 'passed' || first !== row.preflight.first_frame
        || frames !== row.preflight.frames || frames * 4 > MAX_AUDIO_BYTES) {
      throw Error('Hasil cek suara tidak cocok dengan audio awal. Klik Mulai sesi untuk mencoba lagi.');
    }
    Object.assign(row, { first_frame: first, frames, blocks: chunks.length });
    // Keep the profile and the qualifying audio together if saving is interrupted.
    await transaction(['profiles', 'sessions', 'chunks'], 'readwrite', tx => {
      tx.objectStore('profiles').put({ key: 'current', study: row.study });
      tx.objectStore('sessions').put(row);
      for (const chunk of chunks) tx.objectStore('chunks').put({ ...chunk, session_id: row.session_id });
    });
    return { type: 'started', request_id };
  }
  if (type === 'chunk') {
    const row = await getSession(message.session_id);
    if (!row || row.status !== 'recording') throw Error('Sesi tidak sedang merekam.');
    const c = message.chunk;
    if (c.seq !== (row.blocks || 0) || (row.first_frame !== undefined && c.first_frame !== row.first_frame + row.frames)) throw Error('Blok audio hilang atau berulang.');
    if (c.data.length !== c.frames || c.frames < 1) throw Error('Ukuran blok audio tidak valid.');
    row.first_frame ??= c.first_frame; row.frames = (row.frames || 0) + c.frames; row.blocks = (row.blocks || 0) + 1;
    row.events = message.events; row.anchors = message.anchors; row.prompts = message.prompts;
    row.capture_issues = message.capture_issues; row.updated_at = new Date().toISOString();
    await transaction(['sessions', 'chunks'], 'readwrite', tx => {
      tx.objectStore('chunks').put({ ...c, session_id: row.session_id });
      tx.objectStore('sessions').put(row);
    });
    return { type: 'saved', request_id, frames: row.frames };
  }
  if (type === 'finish') {
    const row = await getSession(message.session_id);
    const originalStudy = row.study;
    Object.assign(row, message.snapshot);
    if (row.schema === SCHEMA && row.status === 'complete' && !selectionReport(row.events, true).complete) { row.status = 'interrupted'; row.study = originalStudy; row.capture_issues = [...row.capture_issues, 'quota_incomplete']; }
    try { await put(row); }
    catch { row.status = 'interrupted'; row.study = originalStudy; row.capture_issues = [...row.capture_issues, 'metadata_storage_failed']; }
    volatile.set(row.session_id, row);
    const result = await payload(row.session_id);
    return { type: 'finished', request_id, session: result.session, timing: result.timing };
  }
  if (type === 'export') {
    const result = await payload(message.session_id, true);
    const data = zipSync(result.files, { level: 0, mtime: new Date(result.session.started_at) });
    self.postMessage({ type: 'exported', request_id, data, session: result.session }, [data.buffer]);
    return null;
  }
  if (type === 'restore') {
    if (message.data.byteLength > 160 * 1024 ** 2) throw Error('Paket terlalu besar.');
    const entries = unzipSync(new Uint8Array(message.data), { filter: f => f.name === 'session.json' && f.originalSize < 1024 ** 2 });
    if (!entries['session.json']) throw Error('session.json tidak ditemukan.');
    const session = JSON.parse(decoder.decode(entries['session.json']));
    if (![SCHEMA, LEGACY_SCHEMA].includes(session.schema)) throw Error('Versi paket tidak didukung.');
    return { type: 'restored', request_id, study: validateStudy(session.study) };
  }
  throw Error('Permintaan worker tidak dikenal.');
}
self.onmessage = event => {
  const msg = event.data;
  chain = chain.then(async () => {
    try { const reply = await handle(msg); if (reply) self.postMessage(reply); }
    catch (e) { self.postMessage({ type: 'error', request_id: msg.request_id, message: e.message }); }
  });
};
