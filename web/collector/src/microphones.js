export const REQUESTED_AUDIO = { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: { ideal: 1 }, sampleRate: { ideal: 48000 } };
export function microphoneRequest(device = 'default') {
  return { audio: { ...REQUESTED_AUDIO, ...(device && device !== 'default' ? { deviceId: { exact: device } } : {}) }, video: false };
}
export function populateMicrophones(select, devices) {
  const selected = select.value || 'default';
  select.replaceChildren(new Option('Default perangkat (otomatis)', 'default'));
  for (const device of devices) {
    if (!device.deviceId || device.deviceId === 'default') continue;
    select.append(new Option(device.label || `Mikrofon ${select.options.length}`, device.deviceId));
  }
  select.value = [...select.options].some(o => o.value === selected) ? selected : 'default';
}
