import { test, expect } from '@playwright/test';
import { readFile, mkdir } from 'node:fs/promises';
import { zipSync, unzipSync } from 'fflate';
import { installAudioFixture } from './audio-fixture.mjs';
import { fixturePackage } from '../fixture.mjs';

export async function spaces(page, count = 8) {
  await page.locator('#sync-instruction').focus();
  for (let i = 0; i < count; i++) { await expect(page.locator('#sync-cue-label')).toHaveText('Tekan Spasi sekarang'); await page.keyboard.press('Space'); }
}
async function begin(page, mode = 'clear') {
  await installAudioFixture(page, mode); await page.goto('/');
  await page.locator('#laptop').fill('Synthetic ThinkPad T14');
  await page.locator('#consent').check(); await page.locator('#start').click();
  await expect(page.locator('#phase-badge')).toHaveText('Pemeriksaan awal', { timeout: 15000 });
}
async function prepare(page, mode = 'clear') {
  await begin(page, mode); await spaces(page);
  await expect(page.locator('#phase-badge')).toHaveText('Sedang merekam', { timeout: 15000 });
}
async function download(page, name = '#download-latest') {
  const target = typeof name === 'string' ? page.locator(name) : name;
  const [file] = await Promise.all([page.waitForEvent('download'), target.click()]);
  const bytes = await readFile(await file.path()), files = unzipSync(bytes), decode = value => new TextDecoder().decode(value);
  return { file, bytes, files, session: JSON.parse(decode(files['session.json'])), events: decode(files['events.jsonl']).trim().split('\n').filter(Boolean).map(JSON.parse) };
}
test('full quota, actual key feedback, preserved surplus, ZIP export and profile restore', async ({ page }) => {
  await mkdir('.sites-runtime/verification', { recursive:true });
  const errors = [], requests = []; page.on('pageerror', e => errors.push(e.message));
  page.on('request', r => { if (r.method() !== 'GET') requests.push(r.method()); });
  await prepare(page);
  expect(await page.evaluate(() => window.audioFixtureRequests.length)).toBe(1);
  await expect(page.locator('#audio-wave')).toHaveAttribute('data-active', 'true');
  const sentence = await page.locator('#prompt').textContent(); await page.keyboard.type(sentence, { delay: 8 });
  await expect(page.locator('#prompt')).toHaveAttribute('data-prompt-kind', 'topup');
  const rows = page.locator('#prompt .topup-line');
  const cursor = page.locator('#prompt .cursor');
  const typePrompt = text => page.keyboard.type(text.replaceAll('␣', ' '), { delay: 8 });
  await expect(rows).toHaveCount(2);
  const firstRow = await rows.first().textContent();
  const preview = await rows.nth(1).textContent();
  expect(firstRow.length).toBeGreaterThan(8);
  expect(preview.length).toBe(firstRow.length);
  const promptWidth = (await page.locator('#prompt').boundingBox()).width;
  expect((await rows.first().boundingBox()).width).toBeCloseTo(promptWidth);
  expect((await rows.nth(1).boundingBox()).y).toBeGreaterThan((await rows.first().boundingBox()).y);
  await typePrompt(firstRow.slice(0, 3));
  await expect(cursor).toHaveText(firstRow[3]);
  await page.setViewportSize({ width: 390, height: 850 });
  await expect.poll(() => rows.first().locator('span').count()).toBeLessThan(firstRow.length);
  await expect(cursor).toHaveText(firstRow[3]);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(rows.first().locator('span')).toHaveCount(firstRow.length);
  await typePrompt(firstRow.slice(3));
  await expect(rows.first()).toHaveText(preview);
  await expect(cursor).toHaveText(preview[0]);
  await expect(page.locator('#typing-input')).toBeFocused();
  await page.screenshot({ path:'.sites-runtime/verification/typing-desktop.png', fullPage:true });
  await page.keyboard.type('a'.repeat(30), { delay: 8 }); await page.keyboard.press('Backspace');
  await page.keyboard.press('Tab'); await expect(page.locator('#typing-input')).toBeFocused();
  for (const code of ['ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight', 'AltLeft', 'AltRight', 'ArrowLeft', 'Delete']) {
    await page.keyboard.down(code); await expect(page.locator(`[data-extra-key="${code}"]`)).toHaveClass(/pressed/); await page.keyboard.up(code);
    await expect(page.locator(`[data-extra-key="${code}"] small`)).toHaveText('1 ketukan');
  }
  await page.locator('#typing-input').dispatchEvent('paste');
  const qQuota = page.locator('[data-key="q"] small');
  const qCount = Number((await qQuota.textContent()).split('/')[0]);
  await page.keyboard.down('q'); await expect(page.locator('[data-key="q"]')).toHaveClass(/pressed/); await page.keyboard.up('q');
  await expect(qQuota).toHaveText(`${Math.min(qCount + 1, 25)}/25`);
  for (let i = 0; i < 150; i++) {
    const before = Number(await page.locator('#remaining').textContent());
    if (before === 675) break;
    const suffix = await rows.locator('span:not(.typed)').evaluateAll(nodes => nodes.map(node => node.textContent).join(''));
    expect(suffix.length).toBeGreaterThan(0);
    await typePrompt(suffix);
    await expect(page.locator('#remaining')).toHaveText(String(before + suffix.length));
  }
  await expect(page.locator('#remaining')).toHaveText('675'); await expect(page.locator('#coverage-total')).toHaveText('27 / 27');
  await expect(page.locator('#phase-badge')).toHaveText('Pemeriksaan akhir'); await spaces(page);
  await expect(page.locator('#result')).toBeVisible(); const result = await download(page);
  await result.file.saveAs(test.info().outputPath('session.zip'));
  await mkdir('.sites-runtime/verification', { recursive:true });
  await result.file.saveAs('.sites-runtime/verification/session.zip');
  expect(Object.keys(result.files).sort()).toEqual(['audio.wav','events.jsonl','session.json','timing.json']);
  expect(result.session.schema).toBe('ketik-web-session/v2'); expect(result.session.status).toBe('complete'); expect(result.session.synthetic).toBe(true);
  expect(result.session.preflight.status).toBe('passed'); expect(result.session.collection.selected_total).toBe(675);
  expect(result.session.preflight.method).toBe('key-signal-v3');
  expect(result.session.preflight.source).toBe('session-sync-start');
  expect(result.session.preflight.first_frame).toBe(result.session.first_frame);
  expect(result.session.preflight.markers.map(marker => marker.event_seq)).toEqual(
    result.events.filter(e => e.calibration && e.phase === 'sync_start').map(e => e.seq)
  );
  expect(result.session.preflight.markers.every(m=>m.frame_source==='audio-context-snapshot')).toBe(true);
  expect(result.session.study.setup.device_name).toBe('Synthetic ThinkPad T14');
  expect(result.file.suggestedFilename()).toContain('Synthetic-ThinkPad-T14');
  expect(result.session.additional_keys.counts.ControlLeft).toBe(1); expect(result.session.additional_keys.counts.Tab).toBe(1);
  await expect(page.locator('.history-details strong')).toHaveText('Synthetic ThinkPad T14');
  await expect(page.locator('.history-details span')).toContainText('Kalibrasi');
  expect(Object.values(result.session.collection.selected_counts).every(n => n === 25)).toBe(true);
  expect(result.session.collection.raw_counts.a).toBeGreaterThan(25); expect(result.session.collection.selected_counts.space).toBe(25);
  expect(result.events.filter(e => e.calibration).length).toBe(16);
  const timing=JSON.parse(new TextDecoder().decode(result.files['timing.json']));
  expect(timing.calibration.every(marker=>marker.reason===null)).toBe(true);
  expect(Math.abs(timing.calibration[7].peak_frame-timing.calibration[7].clock_frame)/result.session.sample_rate*1000).toBeLessThan(100);
  expect(result.events.some(e => e.type === 'paste')).toBe(true); expect(result.events.some(e => e.key === 'Backspace')).toBe(true);
  expect(result.events.filter(e => e.quota_selected).length).toBe(675);
  expect(result.events.filter(e => e.calibration).every(e => !e.quota_selected)).toBe(true);
  expect(errors).toEqual([]); expect(requests).toEqual([]);
  expect(Buffer.compare(result.bytes, (await download(page)).bytes)).toBe(0);
  await page.reload(); await expect(page.locator('#session-count')).toHaveText('1 sesi');
  page.on('dialog', d => d.accept()); await page.locator('#restore-file').setInputFiles({ name:'session.zip', mimeType:'application/zip', buffer:result.bytes });
  await expect(page.locator('#mic-feedback')).toContainText('Profil dipulihkan');
  await expect(page.locator('#participant-code')).toHaveText(result.session.participant_id);
});
test('microphones can be listed and selected before the first sound check', async ({ page }) => {
  await installAudioFixture(page); await page.goto('/');
  const typing = await page.locator('#typing-card').boundingBox();
  const setup = await page.locator('#setup-card').boundingBox();
  expect(typing.x + typing.width).toBeLessThan(setup.x);
  await expect(page.locator('#prompt')).toBeVisible();
  await expect(page.locator('#coverage')).toBeVisible();
  await expect(page.locator('#typing-input')).toBeDisabled();
  await expect(page.locator('#microphone')).toBeVisible();
  await expect(page.locator('#microphone')).toBeEnabled(); await expect(page.locator('#microphone')).toHaveValue('default');
  await expect(page.locator('#keyboard')).toHaveCount(0);
  await page.locator('#laptop').fill('Synthetic microphone selection');
  await page.locator('#consent').check(); await page.locator('#load-mics').click();
  await expect(page.locator('#mic-picker-note')).toContainText('Pilih mikrofon');
  expect(await page.evaluate(() => window.audioFixtureTracks.every(t => t.readyState === 'ended'))).toBe(true);
  await expect(page.locator('#phase-badge')).toHaveText('Belum merekam');
  await page.locator('#microphone').selectOption('synthetic-mic-2');
  await page.locator('#start').click();
  await expect(page.locator('#phase-badge')).toHaveText('Hening sebentar', { timeout:15000 });
  expect(await page.evaluate(() => window.audioFixtureRequests[0].audio.deviceId)).toBeUndefined();
  expect(await page.evaluate(() => window.audioFixtureRequests.at(-1).audio.deviceId.exact)).toBe('synthetic-mic-2');
  await page.locator('#cancel').click();
});
test('standalone microphone check shows the result and leaves the study untouched', async ({ page }) => {
  await installAudioFixture(page, 'room-floor');
  await page.goto('/');
  const check = page.getByRole('button', { name: 'Periksa mikrofon', exact: true });
  await check.click();
  await expect(page.locator('#consent')).toBeFocused();
  await expect(page.locator('#mic-feedback')).toContainText('Centang persetujuan');
  expect(await page.evaluate(() => window.audioFixtureRequests.length)).toBe(0);

  await page.locator('#consent').check();
  await expect(page.locator('#laptop')).toHaveValue('');
  await check.click();
  await expect(page.getByRole('heading', { name: 'Periksa mikrofon' })).toBeVisible();
  await expect(page.locator('#phase-badge')).toHaveText('Pemeriksaan awal', { timeout: 15000 });
  await expect(page.locator('#audio-wave')).toHaveAttribute('data-active', 'true');
  await expect(page.locator('#live-level')).toHaveText('-46 dBFS');
  await expect.poll(() => page.locator('#mic-meter').evaluate(meter => meter.value)).toBeGreaterThan(-50);
  await spaces(page, 1);
  await expect(page.locator('#sync-countdown')).toBeVisible();
  await page.keyboard.press('Space');
  await expect(page.locator('#sync-number')).toHaveText('1 / 8');
  await spaces(page, 7);

  await expect(page.locator('#mic-feedback')).toHaveText('Mikrofon siap. 8/8 ketukan terdengar jelas.');
  await expect(check).toBeFocused();
  await expect(page.locator('#start')).toBeEnabled();
  await expect(page.locator('#typing-input')).toBeDisabled();
  await expect(page.locator('#mic-status')).toHaveText('8/8 ketukan jelas');
  await expect(page.locator('#session-count')).toHaveText('0 sesi');
  expect(await page.evaluate(async () => (await import('/storage.js')).getProfile())).toBeNull();
  expect(await page.evaluate(() => window.audioFixtureTracks[0].readyState)).toBe('ended');

  await page.getByText('Rincian pemeriksaan', { exact: true }).click();
  await expect(page.locator('#mic-check-taps [data-result="clear"]')).toHaveCount(8);
  await expect(page.locator('#mic-check-device')).toContainText('Fake deterministic microphone');
  const [file] = await Promise.all([page.waitForEvent('download'), page.locator('#download-check').click()]);
  const report = JSON.parse(await readFile(await file.path(), 'utf8'));
  expect(report.preflight.source).toBe('standalone');
  expect(report.preflight.passed).toBe(8);
  expect(report.marker_events).toHaveLength(8);

  await page.locator('#microphone').selectOption('synthetic-mic-2');
  await expect(page.locator('#mic-check-result')).toBeHidden();
  await expect(page.locator('#download-check')).toBeHidden();
  await check.click();
  await expect(page.locator('#phase-badge')).toHaveText('Hening sebentar', { timeout: 15000 });
  expect(await page.evaluate(() => window.audioFixtureRequests.at(-1).audio.deviceId.exact)).toBe('synthetic-mic-2');
  await page.getByRole('button', { name: 'Batal', exact: true }).click();
  await expect(check).toBeEnabled();
  expect(await page.evaluate(() => window.audioFixtureTracks.every(track => track.readyState === 'ended'))).toBe(true);
  await expect(page.locator('#session-count')).toHaveText('0 sesi');
});
test('standalone microphone check explains weak taps and allows another attempt', async ({ page }) => {
  await installAudioFixture(page, 'weak');
  await page.goto('/');
  await page.locator('#consent').check();
  const check = page.getByRole('button', { name: 'Periksa mikrofon', exact: true });
  await check.click();
  await expect(page.locator('#phase-badge')).toHaveText('Pemeriksaan awal', { timeout: 15000 });
  await spaces(page);
  await expect(page.locator('#mic-feedback')).toContainText('0/8 ketukan jelas. Ketukan kurang terdengar.');
  await expect(page.locator('#mic-feedback')).toContainText('Klik Periksa mikrofon');
  await expect(check).toBeEnabled();
  await expect(page.locator('#session-count')).toHaveText('0 sesi');
});
for (const mode of ['room-floor', 'room-floor-soft']) test(`readiness uses relative sound above the device floor: ${mode}`, async ({ page }) => {
  await prepare(page, mode);
  await expect(page.locator('#microphone')).toHaveValue('default');
  await expect(page.locator('#noise-floor')).toContainText(mode === 'room-floor' ? '-46 dBFS' : '-66 dBFS');
  await expect(page.locator('#tap-results li[data-result="clear"]')).toHaveCount(8);
  expect(await page.evaluate(() => window.audioFixtureRequests.every(r => r.audio.deviceId === undefined))).toBe(true);
});
for (const missingSnapshot of [false,true]) test(`sound assessment with unusable performance anchors, missing snapshot: ${missingSnapshot}`, async ({page}) => {
  await installAudioFixture(page,'room-floor');
  await page.route('**/capture.js',async route=>{
    const response=await route.fetch();
    let source=(await response.text()).replace('const anchor = { before_ms, after_ms, context_s };','const anchor = { before_ms, after_ms: before_ms + 12, context_s };');
    if(missingSnapshot)source=source.replace('return anchor;','return undefined;');
    await route.fulfill({response,body:source});
  });
  await page.goto('/'); await page.locator('#laptop').fill('Synthetic clock regression');
  await page.locator('#consent').check(); await page.locator('#start').click();
  await expect(page.locator('#phase-badge')).toHaveText('Pemeriksaan awal',{timeout:15000});
  await spaces(page,1);
  await expect(page.locator('#tap-results li')).toHaveCount(1,{timeout:5000});
  await expect(page.locator('#tap-results li')).toContainText(missingSnapshot?'Waktu belum cocok':'Jelas');
  await spaces(page,7);
  await expect(page.locator('#tap-results li')).toHaveCount(8);
  await expect(page.locator('#noise-floor')).toContainText('-46 dBFS');
  if(missingSnapshot){
    await expect(page.locator('#mic-feedback')).toContainText('Waktu ketukan belum cocok');
  }else{
    await expect(page.locator('#phase-badge')).toHaveText('Sedang merekam');
    await page.locator('#cancel').click();
  }
  await expect(page.locator('#start')).toBeEnabled();
  const waiting=page.waitForEvent('download'); await page.locator('#download-check').click(); const file=await waiting;
  const result=JSON.parse(await readFile(await file.path(),'utf8'));
  expect(result.schema).toBe('ketik-microphone-check/v1');
  expect(result.preflight.diagnostics.usable_performance_anchors).toBe(0);
  expect(result.preflight.passed).toBe(missingSnapshot?0:8);
  expect(result.preflight.markers.every(m=>missingSnapshot?m.reason==='outside_clock':m.frame_source==='audio-context-snapshot')).toBe(true);
});
test('silence-like microphone cannot pass mandatory signal check', async ({ page }) => {
  await begin(page, 'weak'); await spaces(page);
  await expect(page.locator('#mic-feedback')).toContainText('0/8 ketukan jelas', { timeout:15000 });
  await expect(page.locator('#start')).toBeEnabled(); await expect(page.locator('#mic-feedback')).toContainText('Ketukan kurang terdengar');
  await expect(page.locator('#remaining')).toHaveText('0'); await expect(page.locator('#session-count')).toHaveText('0 sesi');
});
test('clipping prevents signal readiness', async ({ page }) => {
  await begin(page, 'clipping'); await spaces(page);
  await expect(page.locator('#mic-feedback')).toContainText('Suara terlalu keras', { timeout:15000 });
  await expect(page.locator('#start')).toBeEnabled();
});
test('losing focus preserves partial raw audio and never advances the study', async ({ page }) => {
  await prepare(page); await page.keyboard.press('Space'); await page.evaluate(() => window.dispatchEvent(new Event('blur')));
  await expect(page.locator('#result-title')).toContainText('parsial'); const r = await download(page);
  expect(r.session.status).toBe('interrupted'); expect(r.session.capture_issues).toContain('window_blurred'); expect(r.session.study.days.calibration).toBeNull();
  expect(r.session.collection.selected_total).toBe(1);
  expect(r.session.collection.selected_counts.space).toBe(1);
  expect(r.session.frames).toBeGreaterThan(48000);
});
test('reload recovers durable audio as a partial v2 package', async ({ page }) => {
  await prepare(page); page.on('dialog', dialog => dialog.accept()); await page.reload();
  await expect(page.locator('#session-count')).toHaveText('1 sesi'); const r = await download(page, page.locator('.history-actions').getByRole('button', { name:'Unduh ZIP', exact:true }));
  expect(r.session.status).toBe('interrupted'); expect(r.session.capture_issues).toContain('recovered_after_reload'); expect(r.session.study.days.calibration).toBeNull();
  expect(r.session.frames).toBeGreaterThanOrEqual(r.session.preflight.frames);
  expect(r.events.filter(event => event.calibration)).toHaveLength(8);
});
test('a metadata quota failure still permits ZIP rescue', async ({ page }) => {
  await page.route('**/data-worker.js', async route => {
    const response = await route.fetch();
    const source = (await response.text()).replace('const put = row => transaction', "const put = row => { if (row.status !== 'recording') throw new DOMException('quota fixture','QuotaExceededError'); return transaction").replace("tx.objectStore('sessions').put(row));", "tx.objectStore('sessions').put(row)); };");
    await route.fulfill({ response, body:source });
  });
  await prepare(page); await page.evaluate(() => window.dispatchEvent(new Event('blur')));
  await expect(page.locator('#result-title')).toContainText('parsial'); const r = await download(page);
  expect(r.session.capture_issues).toContain('metadata_storage_failed'); expect(r.session.frames).toBeGreaterThan(48000); expect(r.session.study.days.calibration).toBeNull();
});
test('PCM byte guard stops safely without declaring quota completion', async ({ page }) => {
  await page.route('**/capture.js', async route => { const r=await route.fetch(); await route.fulfill({response:r, body:(await r.text()).replace('options.maxBytes ?? MAX_AUDIO_BYTES','options.maxBytes ?? (48000 * 4 * 36)')}); });
  await prepare(page); await expect(page.locator('#result-title')).toContainText('parsial', { timeout:20000 }); const r=await download(page);
  expect(r.session.capture_issues).toContain('audio_byte_limit'); expect(r.session.frames).toBe(48000*36); expect(r.session.collection.complete).toBe(false);
});
test('one start click upgrades a legacy profile and preserves the original ZIP', async ({ page }) => {
  await installAudioFixture(page);
  await page.goto('/'); const files=await fixturePackage(), bytes=zipSync(files,{level:0});
  await page.locator('#restore-file').setInputFiles({name:'legacy.zip',mimeType:'application/zip',buffer:Buffer.from(bytes)});
  await expect(page.locator('#laptop')).toHaveValue('synthetic fixture');
  await expect(page.locator('#participant-code')).toHaveText('11111111-1111-4111-8111-111111111111');
  const decode=value=>new TextDecoder().decode(value), session=JSON.parse(decode(files['session.json']));
  const timing=JSON.parse(decode(files['timing.json'])), events=decode(files['events.jsonl']).trim().split('\n').map(JSON.parse);
  await page.evaluate(async ({session,timing,events,wav})=>{
    const storage=await import('/storage.js'), bytes=Uint8Array.from(atob(wav),c=>c.charCodeAt(0));
    const pcm=new Float32Array(bytes.buffer,44,session.frames);
    await storage.transaction(['sessions','chunks'],'readwrite',tx=>{
      tx.objectStore('sessions').put({...session,events,anchors:timing.anchors,sealed_session:session,sealed_timing:timing});
      tx.objectStore('chunks').put({session_id:session.session_id,seq:0,first_frame:session.first_frame,frames:pcm.length,data:pcm});
    });
  },{session,timing,events,wav:Buffer.from(files['audio.wav']).toString('base64')});
  await page.reload(); await expect(page.locator('#session-count')).toHaveText('1 sesi');
  await expect(page.locator('#check-mic')).toBeEnabled();
  await page.locator('#consent').check(); await page.getByRole('button',{name:'Mulai sesi',exact:true}).click();
  await expect(page.locator('#phase-badge')).toHaveText('Pemeriksaan awal',{timeout:15000});
  await spaces(page); await expect(page.locator('#phase-badge')).toHaveText('Sedang merekam');
  expect(await page.evaluate(() => window.audioFixtureRequests.length)).toBe(1);
  const profile=await page.evaluate(async()=> (await import('/storage.js')).getProfile());
  expect(profile.schema).toBe('ketik-web-study/v2'); expect(profile.participant_id).toBe('11111111-1111-4111-8111-111111111111'); expect(profile.study_id).not.toBe('22222222-2222-4222-8222-222222222222'); expect(profile.days.calibration).toBeNull();
  await page.locator('#cancel').click(); await expect(page.locator('#session-count')).toHaveText('2 sesi');
  const original = page.locator('.history-row').filter({hasText:'33333333'});
  const exported=await download(page,original.getByRole('button',{name:'Unduh ZIP',exact:true}));
  for(const [name,data] of Object.entries(files))expect(Buffer.compare(Buffer.from(exported.files[name]),Buffer.from(data))).toBe(0);
});
test('small screen and reduced motion keep feedback readable', async ({ page }) => {
  await page.emulateMedia({reducedMotion:'reduce'}); await page.setViewportSize({width:390,height:844}); await page.goto('/');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await expect(page.locator('#start')).toBeInViewport();
  await expect(page.locator('#check-mic')).toBeVisible();
  await expect(page.locator('#typing-input')).toBeDisabled();
  await mkdir('.sites-runtime/verification', { recursive:true }); await page.screenshot({path:'.sites-runtime/verification/mobile.png',fullPage:true});
  await page.getByRole('button',{name:'Panduan',exact:true}).click(); await expect(page.locator('#guide')).toBeVisible();
});
test('live waveform follows PCM and cancellation keeps typing unavailable', async ({ page }) => {
  await begin(page);
  await expect(page.locator('#audio-wave')).toHaveAttribute('data-active','true');
  await page.keyboard.type('asdf'.repeat(10),{delay:40});
  const level=await page.locator('#audio-wave').getAttribute('data-level');
  expect(Number(level)).toBeGreaterThan(-70);
  await mkdir('.sites-runtime/verification',{recursive:true}); await page.locator('.audio-panel').screenshot({path:'.sites-runtime/verification/live-waveform.png'});
  await page.locator('#cancel').click(); await expect(page.locator('#audio-wave')).toHaveAttribute('data-active','false'); await expect(page.locator('#live-level')).toHaveText('— dBFS'); await expect(page.locator('#start')).toBeEnabled();
  await expect(page.locator('#session-count')).toHaveText('0 sesi');
});
test('permission denial explains recovery beside the microphone', async ({ page }) => {
  await page.addInitScript(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('denied','NotAllowedError');};});
  await page.goto('/'); await page.locator('#laptop').fill('Synthetic permission test');
  await page.locator('#consent').check(); await page.locator('#start').click();
  await expect(page.locator('#mic-feedback')).toContainText('ikon izin'); await expect(page.locator('#mic-feedback')).toBeInViewport(); await expect(page.locator('#start')).toBeEnabled();
});
test('before consent the click explains the next step without requesting audio', async ({ page }) => {
  await page.addInitScript(()=>{window.micCalls=0;navigator.mediaDevices.getUserMedia=async()=>{window.micCalls++;throw Error('unexpected');};});
  await page.goto('/'); await page.locator('#laptop').fill('Synthetic consent test'); await page.locator('#start').click();
  await expect(page.locator('#mic-feedback')).toContainText('Centang persetujuan'); await expect(page.locator('#consent')).toBeFocused(); expect(await page.evaluate(()=>window.micCalls)).toBe(0);
});
test('pending permission is visible and late streams stop after cancellation', async ({ page }) => {
  await installAudioFixture(page);
  await page.addInitScript(()=>{window.micCalls=0;const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=arg=>{window.micCalls++;return new Promise(resolve=>{window.allowMic=async()=>{const stream=await original(arg);window.lateTracks=stream.getTracks();resolve(stream);};});};});
  await page.goto('/'); await page.locator('#laptop').fill('Synthetic permission test');
  await page.locator('#consent').check(); await page.locator('#start').click();
  await expect(page.locator('#sync-description')).toContainText('Klik Izinkan');
  await expect(page.locator('#start')).toBeDisabled();
  await expect.poll(() => page.evaluate(() => window.micCalls)).toBe(1);
  await page.locator('#cancel').click(); await expect(page.locator('#start')).toBeEnabled();
  await page.evaluate(() => window.allowMic());
  await expect.poll(() => page.evaluate(() => window.lateTracks.every(t => t.readyState === 'ended'))).toBe(true);
  expect(await page.evaluate(() => window.micCalls)).toBe(1);
  await expect(page.locator('#phase-badge')).toHaveText('Belum merekam');
});
