import { cp, mkdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
await rm(`${root}dist`, { recursive: true, force: true });
await mkdir(`${root}dist/vendor`, { recursive: true });
await cp(`${root}src`, `${root}dist`, { recursive: true });
await cp(`${root}node_modules/fflate/esm/browser.js`, `${root}dist/vendor/fflate.js`);
await cp(`${root}node_modules/fflate/LICENSE`, `${root}dist/vendor/fflate-LICENSE.txt`);
console.log('Static site ready: dist/');
