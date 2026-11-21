const DB = 'ketik-collector-v1';
export function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      db.createObjectStore('sessions', { keyPath: 'session_id' });
      db.createObjectStore('chunks', { keyPath: ['session_id', 'seq'] });
      db.createObjectStore('profiles', { keyPath: 'key' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
export async function transaction(stores, mode, run) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(stores, mode); let result;
    tx.oncomplete = () => { db.close(); resolve(result?.result); };
    tx.onerror = tx.onabort = () => { db.close(); reject(tx.error || Error('Penyimpanan browser gagal.')); };
    try { result = run(tx); } catch (e) { tx.abort(); reject(e); }
  });
}
export const saveProfile = study => transaction(['profiles'], 'readwrite', tx => tx.objectStore('profiles').put({ key: 'current', study }));
export async function getProfile() { return (await transaction(['profiles'], 'readonly', tx => tx.objectStore('profiles').get('current')))?.study || null; }
export const allSessions = () => transaction(['sessions'], 'readonly', tx => tx.objectStore('sessions').getAll());
export const getSession = id => transaction(['sessions'], 'readonly', tx => tx.objectStore('sessions').get(id));
export const getChunks = id => transaction(['chunks'], 'readonly', tx => tx.objectStore('chunks').getAll(IDBKeyRange.bound([id, 0], [id, Number.MAX_SAFE_INTEGER])));
export async function removeSession(id) {
  return transaction(['sessions', 'chunks'], 'readwrite', tx => {
    tx.objectStore('sessions').delete(id);
    tx.objectStore('chunks').delete(IDBKeyRange.bound([id, 0], [id, Number.MAX_SAFE_INTEGER]));
  });
}
