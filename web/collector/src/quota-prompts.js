import { CLASSES, PER_CLASS } from './quota.js';
export const PROMPT_VERSION = 'id-balanced-3';
export const SENTENCES = {
  calibration: 'Vina, Xenia, dan Zaki memotret burung hantu sambil membaca buku, lalu menguji papan qwerty di kafe dekat plaza.',
  test1: 'Zaki dan Vira mengajak Xenia berfoto di kafe, lalu membaca buku humor sambil mencoba kuis qwerty di plaza.',
  test2: 'Xenia dan Ziva mengajak Zaki mencoba kuis qwerty, lalu berfoto di kafe sambil membaca buku humor dekat plaza.'
};
function order(value) { let result = 2166136261; for (const c of value) result = Math.imul(result ^ c.charCodeAt(0), 16777619) >>> 0; return result; }
export function sentencePrompt(role) { return { id: `${PROMPT_VERSION}:${role}:sentence`, kind: 'sentence', text: SENTENCES[role] }; }
export function topupPrompt(role, counts, seed, round) {
  const virtual = { ...counts };
  const keys = [];
  const remaining = CLASSES.reduce((total, key) => total + Math.max(0, PER_CLASS - counts[key]), 0);
  for (let i = 0; i < remaining; i++) {
    // Keep the previous sequence of tie breakers as the visible rows roll forward.
    const prefix = `${seed}:${round + Math.floor(i / 8)}:${i % 8}:`;
    let chosen;
    let fewest = PER_CLASS;
    let bestOrder = Infinity;
    for (const key of CLASSES) {
      const count = virtual[key];
      if (count >= PER_CLASS || count > fewest) continue;
      const rank = order(prefix + key);
      if (count < fewest || rank < bestOrder) {
        chosen = key;
        fewest = count;
        bestOrder = rank;
      }
    }
    keys.push(chosen);
    virtual[chosen]++;
  }
  return { id: `${PROMPT_VERSION}:${role}:topup:${round}`, kind: 'topup', keys, text: keys.map(k => k === 'space' ? ' ' : k).join('') };
}
