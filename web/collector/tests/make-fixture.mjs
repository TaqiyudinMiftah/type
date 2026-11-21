import { writeFile } from 'node:fs/promises';
import { zipSync } from 'fflate';
import { fixturePackage } from './fixture.mjs';
await writeFile(process.argv[2], zipSync(await fixturePackage(), { level: 0 }));
console.log('Synthetic fixture exported.');
