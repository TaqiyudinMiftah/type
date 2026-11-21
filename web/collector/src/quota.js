export const COLLECTION_PROTOCOL = 'balanced-27-v1';
export const CLASSES = [...'abcdefghijklmnopqrstuvwxyz', 'space'];
export const PER_CLASS = 25;
export const TOTAL_TARGET = CLASSES.length * PER_CLASS;
export const MAX_AUDIO_BYTES = 128 * 1024 ** 2;
export const collectionConfig = () => ({ protocol: COLLECTION_PROTOCOL, classes: [...CLASSES], per_class: PER_CLASS, selection: 'first-eligible-keydown' });
export function validCollectionConfig(c) { return c?.protocol === COLLECTION_PROTOCOL && c.per_class === PER_CLASS && c.selection === 'first-eligible-keydown' && Array.isArray(c.classes) && c.classes.length === CLASSES.length && c.classes.every((k, i) => k === CLASSES[i]) && Object.keys(c).length === 4; }
export const emptyCounts = () => Object.fromEntries(CLASSES.map(k => [k, 0]));
export const displayKey = key => key === 'space' || key === ' ' ? 'Spasi' : key;
export function eventClass(e) {
  if (e.type !== 'keydown' || e.phase !== 'typing' || e.repeat || e.is_composing || e.ctrl || e.alt || e.meta) return null;
  if (e.key === ' ' && e.code === 'Space') return 'space';
  const key = e.key?.toLowerCase();
  return key?.length === 1 && /^[a-z]$/.test(key) ? key : null;
}
export function assignQuota(row, counts) {
  const key = eventClass(row);
  row.quota_selected = Boolean(key && counts[key] < PER_CLASS);
  if (row.quota_selected) counts[key]++;
  return key;
}
export function selectionReport(events, checkFlags = false) {
  const raw = emptyCounts(), selected = emptyCounts(), sequences = [];
  for (const e of events) {
    const key = eventClass(e), chosen = Boolean(key && selected[key] < PER_CLASS);
    if (checkFlags && ((e.type === 'keydown' && e.quota_selected !== chosen) || (e.type !== 'keydown' && e.quota_selected === true))) throw Error('Pemilihan kuota berbeda dari tombol aktual.');
    if (key) raw[key]++;
    if (chosen) { selected[key]++; sequences.push(e.seq); }
  }
  return { ...collectionConfig(), raw_counts: raw, selected_counts: selected, selected_sequences: sequences, selected_total: sequences.length, complete: CLASSES.every(k => selected[k] === PER_CLASS) };
}
