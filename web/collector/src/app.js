import { APP_VERSION, SCHEMA, ROLES, ROLE_NAMES, newStudy, validateStudy, localDay, nextRole, hash } from './core.js';
import { CLASSES, PER_CLASS, TOTAL_TARGET, emptyCounts, assignQuota, displayKey, selectionReport } from './quota.js';
import { sentencePrompt, topupPrompt, PROMPT_VERSION } from './quota-prompts.js';
import { allSessions, getProfile, saveProfile, removeSession } from './storage.js';
import { Capture } from './capture.js';
import { SignalEnvelope } from './signal.js';
import { AudioFeedback, createKeyboard, createExtraKeyboard } from './feedback.js';
import { EXTRA_KEYS, emptyExtraCounts, extraKey } from './extra-keys.js';
import { deviceName, sessionTitle, sessionFilename, sessionStage } from './session-label.js';
import { REQUESTED_AUDIO, microphoneRequest, populateMicrophones } from './microphones.js';

const $ = id => document.getElementById(id);
const worker = new Worker('./data-worker.js', { type: 'module' });
const pending = new Map();
const state = { study: null, run: null, role: 'calibration', phase: 'idle', working: false, initialized: false, check: null };
const audioUI = new AudioFeedback($('audio-wave'), $('live-level'), $('sound-status'), $('mic-meter'));
const keyboardUI = createKeyboard($('letter-grid'));
const extraKeyboardUI = createExtraKeyboard($('extra-key-grid'));
const SYNC_PAUSE_MS = 3000;
const SYNC_TAPS = 8;
const syncing = () => state.phase === 'sync_start' || state.phase === 'sync_end';
const current = run => state.run === run && !run.ending;
const quotaStudy = study => study?.schema === 'ketik-web-study/v2';
const totalSelected = run => Object.values(run.selected).reduce((a, b) => a + b, 0);
let requestId = 0, phaseTimer, syncTimer, finishTimer, lastExport, paintQueued = false;

function ask(type, body = {}, transfer = []) {
  const request_id = ++requestId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(request_id);
      reject(Error('Penyimpanan tidak merespons. Cek sesi yang tersimpan di browser.'));
    }, 120000);
    pending.set(request_id, { resolve, reject, timer });
    worker.postMessage({ type, request_id, ...body }, transfer);
  });
}

worker.onmessage = ({ data: message }) => {
  const request = pending.get(message.request_id);
  if (!request) return;
  clearTimeout(request.timer);
  pending.delete(message.request_id);
  if (message.type === 'error') request.reject(Error(message.message));
  else request.resolve(message);
};
worker.onerror = () => {
  for (const request of pending.values()) {
    clearTimeout(request.timer);
    request.reject(Error('Penyimpanan rekaman berhenti.'));
  }
  pending.clear();
  interrupt('storage_worker_error');
};

function showMessage(id, text, error = false) {
  const element = $(id);
  element.textContent = text;
  element.classList.toggle('error', error);
  element.hidden = !text;
}

function noticeError(error) {
  showMessage('message', error.message || String(error), true);
}

const supported = Boolean(window.isSecureContext && window.AudioWorkletNode && window.indexedDB
  && navigator.mediaDevices?.getUserMedia && navigator.locks && /Chrome\//.test(navigator.userAgent)
  && !/Android|iPhone|iPad/.test(navigator.userAgent));
if (!supported) showMessage('compatibility', 'Untuk merekam, buka situs ini di Chrome atau Edge desktop.');

function refreshSetup() {
  const working = Boolean(state.run || state.working);
  if (!state.run) {
    try {
      state.role = quotaStudy(state.study) ? nextRole(state.study) : 'calibration';
    } catch (error) {
      state.role = null;
      noticeError(error);
    }
    $('prompt').classList.remove('topup');
    $('prompt').textContent = sentencePrompt(state.role || 'calibration').text;
    const report = state.check?.report;
    $('mic-status').textContent = report ? `${report.passed}/${report.total} ketukan jelas` : 'Belum diperiksa';
  }
  $('study-stage').textContent = state.role
    ? `${ROLE_NAMES[state.role]} · Hari ${ROLES.indexOf(state.role) + 1} dari 3`
    : 'Tiga tahap sudah tercatat';
  $('start-note').textContent = state.role
    ? 'Ikuti petunjuk Spasi, lalu ketik sampai kuota 675 ketukan terpenuhi.'
    : 'Unduh rekaman di Sesi saya atau buat studi baru.';
  $('participant-code').textContent = state.study?.participant_id || 'Dibuat saat sesi pertama';
  $('start').disabled = !supported || !state.initialized || working || !state.role;
  $('check-mic').disabled = !supported || !state.initialized || working;
  for (const id of ['laptop', 'consent', 'load-mics', 'microphone', 'restore-open', 'new-study', 'download-latest', 'download-check']) {
    $(id).disabled = working;
  }
  $('guide-open').disabled = Boolean(state.run);
  $('copy-code').disabled = !state.study || working;
  $('new-study').hidden = !quotaStudy(state.study);
  $('download-check').hidden = !state.check;
  $('mic-check-result').hidden = state.check?.report.source !== 'standalone';
}

function expectedKey(run) {
  if (syncing()) return 'space';
  if (state.phase === 'typing' && run.prompt?.kind === 'topup') return run.topupKeys[run.topupIndex];
  return null;
}

function paint() {
  const run = state.run;
  const selected = run?.selected ?? emptyCounts();
  keyboardUI.update(selected, run ? expectedKey(run) : null);
  extraKeyboardUI.update(run?.extras ?? emptyExtraCounts());
  $('remaining').textContent = String(run ? totalSelected(run) : 0);
  $('coverage-total').textContent = `${CLASSES.filter(key => selected[key] === PER_CLASS).length} / ${CLASSES.length}`;
  if (run && state.phase === 'typing') renderPrompt(run);
}

function queuePaint() {
  if (paintQueued) return;
  paintQueued = true;
  requestAnimationFrame(() => {
    paintQueued = false;
    paint();
  });
}

async function history() {
  const sessions = (await allSessions()).sort((a, b) => b.started_at.localeCompare(a.started_at));
  $('session-count').textContent = `${sessions.length} sesi`;
  const list = $('session-list');
  list.replaceChildren();
  if (!sessions.length) {
    list.innerHTML = '<div class="empty-state"><p>Belum ada sesi tersimpan.</p></div>';
    return;
  }
  for (const row of sessions) {
    const item = document.createElement('div');
    item.className = 'history-row';
    const details = document.createElement('div');
    details.className = 'history-details';
    const title = document.createElement('strong');
    title.textContent = sessionTitle(row);
    const info = document.createElement('span');
    const seconds = Math.round((row.frames || 0) / row.sample_rate);
    info.textContent = `${sessionStage(row)} · ${row.local_day} · ${seconds} detik · ${row.session_id.slice(0, 8)}`;
    details.append(title, info);

    const badge = document.createElement('span');
    badge.className = 'pill';
    badge.textContent = 'Parsial';
    if (row.status === 'complete') {
      badge.textContent = row.sealed_timing?.status === 'checked' ? 'Selesai' : 'Timing perlu ditinjau';
    }
    const actions = document.createElement('div');
    actions.className = 'history-actions';
    const download = document.createElement('button');
    download.className = 'button secondary';
    download.textContent = 'Unduh ZIP';
    download.disabled = !row.frames;
    download.onclick = () => exportSession(row.session_id).catch(noticeError);
    const remove = document.createElement('button');
    remove.className = 'button secondary danger';
    remove.textContent = 'Hapus';
    remove.onclick = () => idleLock(async () => {
      if (!confirm('Hapus salinan sesi ini dari browser?')) return;
      await removeSession(row.session_id);
      if (lastExport === row.session_id) $('result').hidden = true;
      await history();
    }).catch(noticeError);
    actions.append(download, remove);
    item.append(details, badge, actions);
    list.append(item);
  }
}

function setPhase(phase) {
  clearInterval(syncTimer);
  state.phase = phase;
  const labels = {
    idle: 'Belum merekam', opening: 'Menyiapkan mikrofon', quiet: 'Hening sebentar',
    sync_start: 'Pemeriksaan awal', assessing: 'Menilai suara', typing: 'Sedang merekam',
    sync_end: 'Pemeriksaan akhir', finishing: 'Menyimpan rekaman',
  };
  const idle = phase === 'idle';
  const typing = phase === 'typing';
  const finishing = phase === 'finishing';
  const showTyping = idle || typing;
  $('typing-card').dataset.phase = phase;
  $('phase-badge').textContent = labels[phase];
  if (!idle) $('mic-status').textContent = labels[phase];
  $('typing-title').textContent = state.run?.checkOnly ? 'Periksa mikrofon'
    : idle ? 'Mulai dari satu kalimat' : finishing ? labels[phase] : ROLE_NAMES[state.role];
  $('sync-title').textContent = labels[phase];
  $('sync-number').hidden = !syncing();
  $('sync-cue').hidden = !syncing();
  $('sync-instruction').hidden = showTyping || finishing;
  $('prompt').hidden = !showTyping;
  $('typing-toolbar').hidden = !showTyping && phase !== 'sync_end';
  $('audio-panel').hidden = idle || phase === 'opening' || finishing;
  $('cancel').hidden = idle || finishing;
  $('cancel').textContent = state.run?.stored ? 'Hentikan' : 'Batal';
  $('typing-input').hidden = !showTyping;
  $('typing-note').hidden = !showTyping;
  $('typing-input').placeholder = idle ? 'Area mengetik akan aktif setelah mikrofon siap.' : 'Ketik sesuai panduan';
  $('typing-input').disabled = !typing;
  if (typing) $('typing-input').focus();
  else if (!idle && !finishing) $('sync-instruction').focus();
  keyboardUI.clear();
  extraKeyboardUI.clear();
  queuePaint();
}

function nextPrompt(run) {
  const prompt = run.prompts.length
    ? topupPrompt(run.role, run.selected, run.study.study_id, run.prompts.length)
    : sentencePrompt(run.role);
  run.prompt = prompt;
  run.topupIndex = 0;
  if (prompt.kind === 'topup') run.topupKeys = [...prompt.keys];
  $('prompt').dataset.promptId = prompt.id;
  $('prompt').dataset.promptKind = prompt.kind;
  run.prompts.push({ ...prompt, version: PROMPT_VERSION, time_ms: performance.now() });
  $('typing-input').value = '';
  $('task-label').textContent = prompt.kind === 'sentence' ? 'Ketik kalimat berikut' : 'Lengkapi kuota';
  renderPrompt(run);
}

function renderPrompt(run) {
  if (!run.prompt) return;
  const prompt = $('prompt');
  const topup = run.prompt.kind === 'topup';
  prompt.classList.toggle('topup', topup);
  if (topup) {
    renderTopup(run, prompt);
    return;
  }
  const text = run.prompt.text;
  const entered = $('typing-input').value;
  const cursor = entered.length;
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < text.length; i++) {
    const span = document.createElement('span');
    span.textContent = text[i];
    if (i < cursor) span.className = entered[i] === text[i] ? 'typed' : 'mismatch';
    if (i === cursor) span.classList.add('cursor');
    fragment.append(span);
  }
  prompt.replaceChildren(fragment);
}

function topupColumns(prompt, width = prompt.clientWidth) {
  const fontSize = parseFloat(getComputedStyle(prompt).fontSize);
  return Math.max(1, Math.floor(width / (1.35 * fontSize)));
}

function renderTopup(run, prompt) {
  const columns = run.topupColumns ??= topupColumns(prompt);
  const cursor = run.topupIndex;
  const first = Math.floor(cursor / columns) * columns;
  const end = Math.min(first + 2 * columns, run.topupKeys.length);
  const fragment = document.createDocumentFragment();
  prompt.style.setProperty('--topup-columns', columns);
  for (let offset = first; offset < end; offset += columns) {
    const line = document.createElement('span');
    line.className = 'topup-line';
    for (let i = offset; i < Math.min(offset + columns, end); i++) {
      const span = document.createElement('span');
      const key = run.topupKeys[i];
      span.textContent = key === 'space' ? '␣' : key;
      if (key === 'space') span.title = 'Tekan Spasi';
      if (i < cursor) span.className = 'typed';
      else if (i === cursor) span.className = 'cursor';
      line.append(span);
    }
    fragment.append(line);
  }
  prompt.replaceChildren(fragment);
}

new ResizeObserver(([entry]) => {
  const run = state.run;
  if (state.phase !== 'typing' || run?.prompt?.kind !== 'topup') return;
  const columns = topupColumns(entry.target, entry.contentRect.width);
  if (columns === run.topupColumns) return;
  run.topupColumns = columns;
  queuePaint();
}).observe($('prompt'));

function advanceTopup(run, key) {
  if (key === run.topupKeys[run.topupIndex]) {
    run.topupIndex++;
    return;
  }
  // A different key can still fill its quota. Drop its last pending slot.
  const last = run.topupKeys.lastIndexOf(key);
  if (last >= run.topupIndex) run.topupKeys.splice(last, 1);
}

function beginSync(run, phase) {
  clearTimeout(phaseTimer);
  run.syncCount = 0;
  run.syncReadyAt = 0;
  run.syncReleasePending = false;
  run.spaceHeld = false;
  if (phase === 'sync_start') audioUI.setNoiseReference(run.signal.noiseProfile());
  setPhase(phase);
  $('sync-title').textContent = phase === 'sync_start' ? 'Cek suara ketukan' : 'Kuota lengkap. Cek Spasi terakhir';
  $('sync-number').textContent = `0 / ${SYNC_TAPS}`;
  updateSyncCue(run);
  syncTimer = setInterval(() => updateSyncCue(run), 50);
  phaseTimer = setTimeout(() => interrupt('calibration_timeout'), 50000);
}

function updateSyncCue(run) {
  const remaining = Math.max(0, Math.ceil((run.syncReadyAt - performance.now()) / 1000));
  const held = run.syncReleasePending;
  const label = held ? 'Lepaskan Spasi' : remaining ? 'detik lagi' : 'Tekan Spasi sekarang';
  const description = 'Tekan Spasi satu kali setiap kali petunjuk muncul.';
  $('sync-countdown').hidden = held || !remaining;
  $('sync-countdown').textContent = String(remaining);
  if ($('sync-cue-label').textContent !== label) $('sync-cue-label').textContent = label;
  if ($('sync-description').textContent !== description) $('sync-description').textContent = description;
  $('sync-cue').dataset.state = held ? 'release' : remaining ? 'waiting' : 'ready';
}

function settleInitialSync(run) {
  clearTimeout(phaseTimer);
  setPhase('assessing');
  $('sync-description').textContent = 'Membandingkan suara ketukan dengan suara latar.';
  // Wait for the PCM after the last tap to arrive before assessing and saving the prefix.
  run.checkThroughFrame = Math.ceil((run.capture.context.currentTime + .4) * run.rate);
  phaseTimer = setTimeout(() => interrupt('audio_check_timeout'), 5000);
}

function scheduleQuotaFinish(run) {
  if (totalSelected(run) !== TOTAL_TARGET || run.held.size || run.quotaFinishing) return;
  run.quotaFinishing = true;
  finishTimer = setTimeout(() => {
    run.quotaFinishing = false;
    if (current(run) && state.phase === 'typing' && !run.held.size) beginSync(run, 'sync_end');
  }, 400);
}

function keyVisual(event) {
  if (event.code === 'Space') return 'space';
  if (/^[a-z]$/i.test(event.key)) return event.key.toLowerCase();
  return Object.hasOwn(EXTRA_KEYS, event.code) ? event.code : null;
}

function recordEvent(event) {
  const run = state.run;
  if (!run || !current(run) || (!syncing() && state.phase !== 'typing')) return;
  if (run.checkOnly && event.code !== 'Space' && event.key !== 'Escape') return;
  const handler = performance.now();
  const anchor = run.capture.anchor();
  const row = {
    seq: run.events.length, type: event.type, phase: state.phase,
    time_ms: event.timeStamp, handler_ms: handler, is_trusted: event.isTrusted,
  };
  let recordedKey;
  if (anchor && Number.isFinite(anchor.context_s)) row.audio_context_s = anchor.context_s;
  if (event.type === 'keydown' || event.type === 'keyup') {
    Object.assign(row, {
      key: event.key, code: event.code, repeat: event.repeat, is_composing: event.isComposing,
      shift: event.shiftKey, ctrl: event.ctrlKey, alt: event.altKey, meta: event.metaKey,
    });
    const visual = keyVisual(event);
    const down = event.type === 'keydown';
    if (visual) {
      keyboardUI.press(visual, down);
      extraKeyboardUI.press(visual, down);
    }
    if (down) {
      run.held.add(event.code);
      $('last-key').textContent = `Tombol terakhir ${EXTRA_KEYS[visual] || displayKey(visual || event.key)}`;
      recordedKey = assignQuota(row, run.selected);
      const extra = extraKey(row);
      if (extra) run.extras[extra]++;
    } else run.held.delete(event.code);
  } else {
    Object.assign(row, {
      input_type: event.inputType || null, data: event.data ?? null,
      selection_start: $('typing-input').selectionStart, selection_end: $('typing-input').selectionEnd,
    });
  }
  if (state.phase === 'typing') row.collection_mode = run.prompt.kind;
  run.events.push(row);

  if (event.type === 'keydown' && event.key === 'Escape') {
    event.preventDefault();
    interrupt('participant_stopped');
    return;
  }
  if (syncing()) {
    if (event.type === 'keydown') {
      if (event.code !== 'Tab') event.preventDefault();
      if (event.code === 'Space') {
        if (!event.repeat && !run.spaceHeld && !event.isComposing && handler >= run.syncReadyAt) {
          row.calibration = true;
          if (state.phase === 'sync_start') run.pendingTaps.push(row);
          run.syncCount++;
          run.syncReleasePending = true;
          $('sync-number').textContent = `${run.syncCount} / ${SYNC_TAPS}`;
          updateSyncCue(run);
        }
        run.spaceHeld = true;
      }
    } else if (event.type === 'keyup' && event.code === 'Space') {
      run.spaceHeld = false;
      if (run.syncReleasePending) {
        run.syncReleasePending = false;
        if (run.syncCount === SYNC_TAPS) {
          if (state.phase === 'sync_start') settleInitialSync(run);
          else finish(run, 'complete').catch(noticeError);
        } else {
          run.syncReadyAt = handler + SYNC_PAUSE_MS;
          updateSyncCue(run);
        }
      }
    }
    return;
  }

  if (event.type === 'keydown') {
    if (event.code === 'Tab' || /^(Control|Alt)/.test(event.code)) event.preventDefault();
    if (run.prompt.kind === 'topup') {
      event.preventDefault();
      if (row.quota_selected) advanceTopup(run, recordedKey);
    } else if (event.code === 'Enter' && !event.repeat) {
      event.preventDefault();
      nextPrompt(run);
    }
    if (!event.repeat && keyVisual(event)) run.livePending.push(row);
  } else if (event.type === 'input' && run.prompt.kind === 'sentence'
             && $('typing-input').value.length >= run.prompt.text.length) {
    nextPrompt(run);
  } else if (event.type === 'keyup') {
    if (totalSelected(run) === TOTAL_TARGET) scheduleQuotaFinish(run);
  }
  queuePaint();
}

for (const type of ['keydown', 'keyup', 'beforeinput', 'input', 'compositionstart', 'compositionend']) {
  $('typing-input').addEventListener(type, recordEvent);
}
for (const type of ['keydown', 'keyup']) $('sync-instruction').addEventListener(type, recordEvent);
for (const type of ['paste', 'drop']) {
  $('typing-input').addEventListener(type, event => {
    event.preventDefault();
    recordEvent(event);
    audioUI.setStatus('Ketik langsung dengan keyboard.', 'warning');
  });
}

function signalStatus(report) {
  if (Number.isFinite(report.snr_db)) {
    const db = report.snr_db;
    $('tap-level').textContent = `Ketukan terakhir ${db >= 0 ? '+' : ''}${db.toFixed(1)} dB terhadap latar`;
  }
  const reasons = {
    clipping: ['Suara terlalu keras. Turunkan level mikrofon.', 'error'],
    weak_signal: ['Ketukan kurang terdengar. Coba dekatkan mikrofon ke keyboard.', 'warning'],
    outside_clock: ['Waktu ketukan belum cocok dengan audio.', 'warning'],
    outside_audio: ['Audio pada posisi ketukan belum lengkap.', 'warning'],
    baseline_incomplete: ['Suara latar belum terukur. Tetap hening saat cek awal.', 'warning'],
  };
  return reasons[report.reason] || ['Ketukan terdengar jelas.', 'good'];
}

function renderTapResults(markers, list = $('tap-results')) {
  const names = {
    weak_signal: 'Kurang terdengar', clipping: 'Terlalu keras', outside_clock: 'Waktu belum cocok',
    outside_audio: 'Audio belum lengkap', baseline_incomplete: 'Latar belum lengkap',
  };
  list.replaceChildren();
  for (const [index, marker] of markers.entries()) {
    const item = document.createElement('li');
    item.dataset.result = marker.reason || 'clear';
    const level = Number.isFinite(marker.snr_db)
      ? ` · ${marker.snr_db >= 0 ? '+' : ''}${marker.snr_db.toFixed(1)} dB`
      : '';
    item.textContent = `${index + 1}. ${names[marker.reason] || 'Jelas'}${level}`;
    list.append(item);
  }
  list.hidden = !markers.length;
}

function updateTapFeedback(run) {
  const preview = state.phase === 'sync_start';
  if (!preview && state.phase !== 'typing') return;
  const queue = preview ? run.pendingTaps : run.livePending;
  while (queue.length) {
    const event = queue[0];
    const signal = run.signal;
    const frame = event.audio_context_s * signal.rate - signal.firstFrame;
    const needsAudio = Number.isFinite(frame)
      ? frame + .25 * signal.rate > signal.frames
      : performance.now() - event.handler_ms < 1500;
    if (needsAudio) break;
    queue.shift();
    const result = signal.tap(event, run.capture.anchors);
    if (preview) {
      run.tapResults.push(result);
      renderTapResults(run.tapResults);
    }
    audioUI.setStatus(...signalStatus(result));
  }
}

function storeChunk(run, chunk) {
  const data = {
    session_id: run.id, chunk, events: run.events, anchors: run.capture.anchors,
    prompts: run.prompts, capture_issues: run.issues,
  };
  ask('chunk', data, [chunk.data.buffer]).catch(error => {
    if (state.run === run) interrupt(`storage: ${error.message}`);
  });
}

function onAudioBlock(run, block) {
  if (state.run !== run || (run.ending && !run.stored)) return;
  run.signal ??= new SignalEnvelope(run.capture.context.sampleRate);
  run.signal.add(block);
  if (!run.checkOnly) {
    const chunk = {
      seq: run.chunkSeq++, first_frame: block.first_frame, frames: block.frames, data: block.data,
    };
    if (run.stored) storeChunk(run, chunk);
    else run.buffered.push(chunk);
  }
  if (!current(run) || run.issues.length) return;
  updateTapFeedback(run);
  if (run.checkThroughFrame && block.first_frame + block.frames >= run.checkThroughFrame) {
    run.checkThroughFrame = null;
    completeCheck(run).catch(error => returnToSetup(run, error.message));
  }
}

async function completeCheck(run) {
  clearTimeout(phaseTimer);
  const report = {
    ...run.signal.assess(run.events, run.capture.anchors),
    checked_at: new Date().toISOString(), microphone_label: run.trackSettings.microphone_label,
    source: run.checkOnly ? 'standalone' : 'session-sync-start',
  };
  state.check = { report, markers: run.events.filter(event => event.calibration) };
  renderTapResults(report.markers);
  if (run.checkOnly) {
    $('mic-check-device').textContent = report.microphone_label || 'Mikrofon perangkat';
    renderTapResults(report.markers, $('mic-check-taps'));
  }
  if (report.status !== 'passed') {
    const failed = report.markers.find(marker => marker.reason);
    const reason = failed ? signalStatus(failed)[0] : 'Pemeriksaan Spasi belum lengkap.';
    await returnToSetup(run, `${report.passed}/${SYNC_TAPS} ketukan jelas. ${reason} Klik ${run.action.textContent} untuk mencoba lagi.`);
    return;
  }
  if (run.checkOnly) {
    await returnToSetup(run, `Mikrofon siap. ${report.passed}/${SYNC_TAPS} ketukan terdengar jelas.`, false);
    return;
  }
  run.saving = true;
  const row = {
    schema: SCHEMA, app_version: APP_VERSION, session_id: run.id, study: run.study, role: run.role,
    local_day: run.day, started_at: run.startedAt, updated_at: new Date().toISOString(),
    status: 'recording', synthetic: /fake|synthetic/i.test(run.trackSettings.microphone_label),
    time_origin_epoch_ms: performance.timeOrigin, sample_rate: run.rate,
    requested_settings: REQUESTED_AUDIO, track_settings: run.trackSettings,
    consent: { version: 'local-audio-key-events-v1', accepted_at: run.consentAt },
    preflight: report, events: run.events, anchors: run.capture.anchors, prompts: run.prompts,
    capture_issues: run.issues, warmup_discarded_frames: run.capture.warmupDiscarded || 0,
  };
  const chunks = run.buffered.splice(0);
  await ask('start', { row, chunks }, chunks.map(chunk => chunk.data.buffer));
  state.study = run.study;
  run.saving = false;
  run.stored = true;
  for (const chunk of run.buffered.splice(0)) storeChunk(run, chunk);
  if (run.issues.length) {
    await finish(run, 'interrupted');
    return;
  }
  setPhase('typing');
  nextPrompt(run);
  audioUI.setStatus('Mikrofon siap. Mulai mengetik.', 'good');
}

function microphoneError(error) {
  const reasons = {
    NotAllowedError: 'Buka ikon izin di sebelah alamat web dan izinkan mikrofon, lalu coba lagi.',
    NotFoundError: 'Tidak ada mikrofon yang terdeteksi. Sambungkan mikrofon, lalu coba lagi.',
    NotReadableError: 'Mikrofon tidak dapat dibuka. Tutup aplikasi lain yang menggunakannya, lalu coba lagi.',
    OverconstrainedError: 'Mikrofon pilihan tidak tersedia. Pilih mikrofon lain dari daftar.',
    SecurityError: 'Akses mikrofon dibatasi. Buka situs ini langsung di Chrome atau Edge desktop.',
  };
  return reasons[error.name] || error.message;
}

async function acquireLock(run) {
  await new Promise((resolve, reject) => {
    navigator.locks.request('ketik-capture', { ifAvailable: true }, lock => {
      if (!lock) {
        reject(Error('Sesi masih aktif di tab lain. Selesaikan sesi itu dahulu.'));
        return;
      }
      if (!current(run)) {
        resolve();
        return;
      }
      return new Promise(release => {
        run.releaseLock = release;
        resolve();
      });
    }).catch(reject);
  });
}

function setupFields(previous) {
  const name = $('laptop').value.trim();
  return {
    device_name: name, laptop: name,
    keyboard: previous?.setup.keyboard || 'Tidak dirinci (kolom perangkat gabungan)',
  };
}

async function start(checkOnly = false) {
  const action = $(checkOnly ? 'check-mic' : 'start');
  if (action.disabled || state.run || state.working) return;
  showMessage('message', '');
  if (!checkOnly && !$('laptop').value.trim()) {
    showMessage('mic-feedback', 'Isi nama perangkat dahulu.', true);
    $('laptop').focus();
    return;
  }
  if (!$('consent').checked) {
    showMessage('mic-feedback', checkOnly ? 'Centang persetujuan untuk memeriksa mikrofon.' : 'Centang persetujuan untuk mulai merekam.', true);
    $('consent').focus();
    return;
  }

  const run = {
    checkOnly, action, id: crypto.randomUUID(), consentAt: new Date().toISOString(), events: [], prompts: [],
    issues: [], selected: emptyCounts(), extras: emptyExtraCounts(), held: new Set(),
    pendingTaps: [], tapResults: [], livePending: [], buffered: [], chunkSeq: 0,
  };
  state.run = run;
  state.check = null;
  if (!checkOnly) $('result').hidden = true;
  $('typing-input').value = '';
  $('last-key').textContent = 'Belum ada tombol ditekan.';
  $('tap-level').textContent = '';
  renderTapResults([]);
  audioUI.setNoiseReference(null);
  setPhase('opening');
  $('sync-description').textContent = 'Klik Izinkan jika browser meminta akses mikrofon.';
  refreshSetup();
  showMessage('mic-feedback', '');

  try {
    await acquireLock(run);
    if (!current(run)) return;
    if (!checkOnly) {
      const existing = await getProfile();
      if (!current(run)) return;
      const previous = existing ? validateStudy(existing) : null;
      const setup = setupFields(previous);
      run.study = quotaStudy(previous) ? previous : newStudy(setup, previous?.participant_id);
      if (quotaStudy(previous)) run.study.setup.device_name = setup.device_name;
    }

    run.permissionTimer = setTimeout(() => {
      if (current(run)) $('sync-description').textContent = 'Masih menunggu izin. Periksa ikon mikrofon di bilah alamat browser.';
    }, 10000);
    const stream = await navigator.mediaDevices.getUserMedia(microphoneRequest($('microphone').value));
    clearTimeout(run.permissionTimer);
    if (!current(run)) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    run.stream = stream;
    const track = stream.getAudioTracks()[0];
    if (!track) throw Error('Tidak ada input mikrofon yang tersedia.');
    run.trackSettings = { microphone_label: track.label };
    if (!checkOnly) {
      if (run.study.setup.microphone && run.study.setup.microphone !== track.label) {
        await refreshDevices(true).catch(() => {});
        throw Error(`Pilih ${run.study.setup.microphone} dari daftar mikrofon untuk melanjutkan studi ini.`);
      }
      run.study.setup.microphone = track.label;
      const settings = track.getSettings();
      const deviceHash = await hash(new TextEncoder().encode(`${run.study.participant_id}:${settings.deviceId || ''}`));
      if (!current(run)) return;
      delete settings.deviceId;
      delete settings.groupId;
      run.trackSettings = { ...settings, microphone_label: track.label, device_id_hash: deviceHash };
      run.startedAt = new Date().toISOString();
      run.day = localDay(run.startedAt, run.study.timezone);
      run.role = nextRole(run.study, run.day);
      if (!run.role) throw Error('Studi ini sudah selesai. Klik Buat studi baru untuk melanjutkan.');
      state.role = run.role;
    }
    setPhase('quiet');
    $('sync-description').textContent = 'Jangan mengetik selama suara latar diukur.';
    audioUI.setStatus('Mengukur suara latar.');
    run.capture = new Capture(stream, block => onAudioBlock(run, block), issue => {
      if (state.run === run) interrupt(issue);
    });
    run.rate = await run.capture.start();
    if (!current(run)) return;
    audioUI.connect(run.capture);
    phaseTimer = setTimeout(() => beginSync(run, 'sync_start'), 3000);
  } catch (error) {
    if (current(run)) await returnToSetup(run, microphoneError(error));
  }
}

function clearRunTimers(run) {
  clearTimeout(run.permissionTimer);
  clearTimeout(phaseTimer);
  clearTimeout(finishTimer);
  clearInterval(syncTimer);
}

function stopCapture(run) {
  if (run.capture) {
    run.stopPromise ??= run.capture.stop();
    return run.stopPromise;
  }
  run.stream?.getTracks().forEach(track => track.stop());
}

function resetRun(run) {
  run.releaseLock?.();
  run.buffered.length = 0;
  audioUI.disconnect();
  state.run = null;
  $('typing-input').value = '';
  $('task-label').textContent = 'Kalimat';
  $('last-key').textContent = 'Belum ada tombol ditekan.';
  setPhase('idle');
  refreshSetup();
}

async function returnToSetup(run, reason, error = true) {
  if (!current(run)) return;
  run.ending = true;
  clearRunTimers(run);
  try {
    await stopCapture(run);
  } catch {
    run.stream?.getTracks().forEach(track => track.stop());
  } finally {
    resetRun(run);
    showMessage('mic-feedback', reason, error);
    if (!document.hidden) run.action.focus();
  }
}

function interrupt(issue) {
  const run = state.run;
  if (!run) return;
  if (!run.issues.includes(issue)) run.issues.push(issue);
  if (run.ending) return;
  if (run.stored) {
    finish(run, 'interrupted').catch(noticeError);
  } else if (run.saving) {
    // Let the local transaction settle before closing a newly saved partial session.
    stopCapture(run)?.catch(() => {});
  } else {
    const reasons = {
      participant_stopped: '',
      calibration_timeout: 'Cek Spasi habis waktu.',
      audio_check_timeout: 'Audio belum lengkap.',
      tab_hidden: 'Pemeriksaan berhenti saat tab ditinggalkan.',
      window_blurred: 'Pemeriksaan berhenti saat jendela kehilangan fokus.',
    };
    const reason = reasons[issue] ?? 'Mikrofon terputus. Periksa sambungan.';
    const message = reason ? `${reason} Klik ${run.action.textContent} untuk mengulang.` : '';
    returnToSetup(run, message).catch(noticeError);
  }
}

async function finish(run, status) {
  if (!current(run)) return;
  run.ending = true;
  clearRunTimers(run);
  setPhase('finishing');
  try {
    if (status === 'complete') await new Promise(resolve => setTimeout(resolve, 500));
    await stopCapture(run);
    if (run.issues.length || !selectionReport(run.events).complete) status = 'interrupted';
    const study = structuredClone(run.study);
    if (status === 'complete') study.days[run.role] ??= run.day;
    const result = await ask('finish', {
      session_id: run.id,
      snapshot: {
        study, status, finished_at: new Date().toISOString(), events: run.events,
        anchors: run.capture.anchors, prompts: run.prompts, capture_issues: run.issues,
        warmup_discarded_frames: run.capture.warmupDiscarded || 0,
      },
    });
    const session = result.session;
    if (session.status === 'complete') {
      await saveProfile(study);
      state.study = study;
    }
    lastExport = run.id;
    $('result').hidden = false;
    $('result-title').textContent = session.status === 'complete' ? 'Sesi tersimpan' : 'Sesi parsial tersimpan';
    $('result-description').textContent = `${sessionTitle(session)} · ${ROLE_NAMES[run.role]} · ${session.collection.selected_total}/${TOTAL_TARGET} ketukan`;
    $('result-quality').textContent = result.timing.status === 'checked' ? '' : 'Timing perlu ditinjau. Audio dan tombol tersedia dalam ZIP.';
  } catch (error) {
    noticeError(error);
  } finally {
    resetRun(run);
    await history();
    if (!$('result').hidden && !document.hidden) $('download-latest').focus();
  }
}

async function idleLock(action) {
  if (state.run || state.working) throw Error('Selesaikan sesi aktif dahulu.');
  const run = async lock => {
    if (navigator.locks && !lock) throw Error('Sesi atau ekspor masih aktif di tab lain.');
    state.working = true;
    refreshSetup();
    try {
      return await action();
    } finally {
      state.working = false;
      refreshSetup();
    }
  };
  return navigator.locks ? navigator.locks.request('ketik-capture', { ifAvailable: true }, run) : run(null);
}

function downloadFile(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

async function exportSession(id) {
  return idleLock(async () => {
    showMessage('message', 'Menyiapkan ZIP…');
    const { data, session } = await ask('export', { session_id: id });
    downloadFile(new Blob([data], { type: 'application/zip' }), sessionFilename(session));
    showMessage('message', 'Unduhan ZIP dimulai.');
    await history();
  });
}

async function refreshDevices(whileOpen = false) {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === 'audioinput');
  if (whileOpen || (!state.run && !state.working)) populateMicrophones($('microphone'), devices);
}

async function listMicrophones() {
  if (state.run || state.working) return;
  if (!$('consent').checked) {
    showMessage('mic-feedback', 'Centang persetujuan untuk mengakses daftar mikrofon.', true);
    $('consent').focus();
    return;
  }
  if (!supported) return;
  state.working = true;
  refreshSetup();
  showMessage('mic-picker-note', 'Izinkan mikrofon jika diminta browser.');
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(microphoneRequest());
    if (document.hidden) return;
    await refreshDevices(true);
    showMessage('mic-picker-note', 'Pilih mikrofon untuk pemeriksaan dan perekaman.');
  } catch (error) {
    showMessage('mic-picker-note', microphoneError(error), true);
  } finally {
    stream?.getTracks().forEach(track => track.stop());
    state.working = false;
    refreshSetup();
  }
}

$('start').onclick = () => start().catch(noticeError);
$('check-mic').onclick = () => start(true).catch(noticeError);
$('cancel').onclick = () => interrupt('participant_stopped');
$('load-mics').onclick = () => listMicrophones().catch(noticeError);
$('microphone').onchange = () => {
  state.check = null;
  refreshSetup();
  showMessage('mic-feedback', '');
};
$('laptop').oninput = () => showMessage('mic-feedback', '');
$('consent').onchange = () => showMessage('mic-feedback', '');
$('download-latest').onclick = () => exportSession(lastExport).catch(noticeError);
$('guide-open').onclick = () => $('guide').showModal();
$('guide-close').onclick = () => $('guide').close();
$('copy-code').onclick = async () => {
  try {
    if (!state.study) return;
    await navigator.clipboard.writeText(state.study.participant_id);
    showMessage('message', 'ID peserta disalin.');
  } catch (error) {
    noticeError(error);
  }
};
$('download-check').onclick = () => {
  if (!state.check || state.run) return;
  const { report, markers } = state.check;
  const diagnostic = {
    schema: 'ketik-microphone-check/v1', app_version: APP_VERSION, preflight: report,
    marker_events: markers.map(event => ({
      seq: event.seq, time_ms: event.time_ms, handler_ms: event.handler_ms, audio_context_s: event.audio_context_s,
    })),
    scope: 'Ringkasan pemeriksaan suara tanpa rekaman atau isi ketikan.',
  };
  const blob = new Blob([JSON.stringify(diagnostic, null, 2) + '\n'], { type: 'application/json' });
  downloadFile(blob, `ketik-pemeriksaan-${report.checked_at.replace(/[:.]/g, '-')}.json`);
};
$('restore-open').onclick = () => $('restore-file').click();
$('restore-file').onchange = async event => {
  try {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > 160 * 1024 ** 2) throw Error('Paket terlalu besar.');
    await idleLock(async () => {
      const data = await file.arrayBuffer();
      const { study } = await ask('restore', { data }, [data]);
      if (state.study && !confirm('Gunakan profil dari ZIP? Sesi yang sudah tersimpan tetap tersedia.')) return;
      await saveProfile(study);
      state.study = study;
      $('laptop').value = deviceName(study);
      showMessage('mic-feedback', 'Profil dipulihkan. Klik Mulai sesi.');
    });
  } catch (error) {
    noticeError(error);
  } finally {
    $('restore-file').value = '';
  }
};
$('new-study').onclick = () => idleLock(async () => {
  if (!$('laptop').value.trim()) {
    showMessage('mic-feedback', 'Isi nama perangkat dahulu.', true);
    $('laptop').focus();
    return;
  }
  if (!confirm('Mulai kalibrasi untuk studi baru? Riwayat sesi tetap tersedia.')) return;
  const study = newStudy(setupFields(state.study), state.study?.participant_id);
  await saveProfile(study);
  state.study = study;
  showMessage('mic-feedback', 'Studi siap. Klik Mulai sesi.');
}).catch(noticeError);

document.addEventListener('visibilitychange', () => {
  if (document.hidden) interrupt('tab_hidden');
});
window.addEventListener('blur', () => {
  if (state.phase !== 'opening') interrupt('window_blurred');
});
window.addEventListener('pagehide', () => {
  state.run?.stream?.getTracks().forEach(track => track.stop());
  audioUI.disconnect();
});
window.addEventListener('beforeunload', event => {
  if (state.run?.stored || state.run?.saving) event.preventDefault();
});
navigator.mediaDevices?.addEventListener('devicechange', () => refreshDevices().catch(() => {}));

try {
  const study = await getProfile();
  if (study) {
    state.study = validateStudy(study);
    $('laptop').value = deviceName(study);
  }
  await history();
  state.initialized = true;
  refreshSetup();
  refreshDevices().catch(() => {});
} catch (error) {
  noticeError(Error(`Penyimpanan lokal belum siap. ${error.message}`));
}
