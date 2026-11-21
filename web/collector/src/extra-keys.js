// Additional physical keys are retained separately from the frozen 27-class quota.
export const EXTRA_KEYS = {
  Backspace: 'Backspace', Enter: 'Enter', Tab: 'Tab', Delete: 'Delete',
  ShiftLeft: 'Shift kiri', ShiftRight: 'Shift kanan',
  ControlLeft: 'Ctrl kiri', ControlRight: 'Ctrl kanan',
  AltLeft: 'Alt kiri', AltRight: 'Alt kanan',
  ArrowLeft: 'Panah kiri', ArrowRight: 'Panah kanan', ArrowUp: 'Panah atas', ArrowDown: 'Panah bawah'
};
export const emptyExtraCounts = () => Object.fromEntries(Object.keys(EXTRA_KEYS).map(k => [k, 0]));
export function extraKey(event) {
  return event.type === 'keydown' && event.phase === 'typing' && !event.repeat && !event.is_composing && Object.hasOwn(EXTRA_KEYS, event.code) ? event.code : null;
}
export function additionalCounts(events) {
  const counts = emptyExtraCounts();
  for (const event of events) { const key = extraKey(event); if (key) counts[key]++; }
  return counts;
}
