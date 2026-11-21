// Test-only generated PCM. No physical microphone or preflight bypass.
export async function installAudioFixture(page, mode = 'clear') {
  await page.addInitScript(({ mode }) => {
    window.audioFixtureTracks = []; window.audioFixtureMode = mode; window.audioFixtureRequests = [];
    let current;
    navigator.mediaDevices.enumerateDevices = async () => [{ kind: 'audioinput', deviceId: 'synthetic-mic', groupId: 'synthetic-group', label: 'Fake deterministic microphone' }, { kind: 'audioinput', deviceId: 'synthetic-mic-2', groupId: 'synthetic-group-2', label: 'Fake second microphone' }];
    navigator.mediaDevices.getUserMedia = async request => {
      window.audioFixtureRequests.push(request);
      const context = new AudioContext({ sampleRate: 48000 }); await context.resume();
      const destination = context.createMediaStreamDestination(), track = destination.stream.getAudioTracks()[0];
      const selected = request?.audio?.deviceId?.exact || 'synthetic-mic';
      Object.defineProperty(track, 'label', { value: selected === 'synthetic-mic-2' ? 'Fake second microphone' : 'Fake deterministic microphone' });
      track.getSettings = () => ({ deviceId: selected, groupId: 'synthetic-group', sampleRate: 48000, channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false });
      if (mode.startsWith('room-floor')) {
        const oscillator = context.createOscillator(), gain = context.createGain();
        oscillator.frequency.value = 1000; gain.gain.value = Math.SQRT2 * (mode === 'room-floor-soft' ? .0005 : .005);
        oscillator.connect(gain); gain.connect(destination); oscillator.start();
      }
      const stop = track.stop.bind(track); track.stop = () => { stop(); context.close().catch(() => {}); };
      window.audioFixtureTracks.push(track); current = { context, destination };
      return destination.stream;
    };
    document.addEventListener('keydown', event => {
      if (!['typing-input', 'sync-instruction'].includes(event.target.id) || !current || current.context.state !== 'running' || event.repeat) return;
      const amplitude = window.audioFixtureMode === 'clipping' ? 1.2 : window.audioFixtureMode === 'weak' ? 1e-6 : window.audioFixtureMode.startsWith('room-floor') ? (window.audioFixtureMode === 'room-floor-soft' ? .004 : .04) : .45;
      const buffer = current.context.createBuffer(1, 240, 48000), data = buffer.getChannelData(0);
      // A short rectangular transient, independent of the collector's peak logic.
      for (let i = 102; i < 138; i++) data[i] = amplitude;
      const source = current.context.createBufferSource(); source.buffer = buffer; source.connect(current.destination); source.start(current.context.currentTime + .035);
    }, true);
  }, { mode });
}
