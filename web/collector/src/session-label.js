import { ROLE_NAMES } from './core.js';

export function deviceName(study) { return study?.setup?.device_name?.trim() || study?.setup?.laptop?.trim() || study?.setup?.keyboard?.trim() || 'Perangkat tanpa nama'; }
export function sessionTitle(session) { return deviceName(session.study); }
export function sessionFilename(session) {
  const name = sessionTitle(session).normalize('NFKC').replace(/[^\p{L}\p{N}._-]+/gu, '-').replace(/^[._-]+|[._-]+$/g, '').slice(0, 80) || 'perangkat';
  return `ketik-${name}-${session.local_day}-${session.role}-${session.session_id.slice(0, 8)}.zip`;
}
export function sessionStage(session) { return ROLE_NAMES[session.role] || session.role; }
