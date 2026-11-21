import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
test('only a contiguous initial block starts the recording; later frame gaps are reported', async () => {
  const messages = [];
  const context = vm.createContext({ sampleRate: 256, currentFrame: 0, Float32Array,
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage: m => messages.push(m) }; } },
    registerProcessor: (_, cls) => { context.Recorder = cls; }
  });
  vm.runInContext(await readFile(new URL('../src/audio-worklet.js', import.meta.url), 'utf8'), context);
  const recorder = new context.Recorder(), input = [[new Float32Array(128)]];
  recorder.process(input, []); context.currentFrame = 1024; recorder.process(input, []);
  context.currentFrame = 1152; recorder.process(input, []);
  const first = messages.find(m => m.type === 'audio');
  assert.equal(first.first_frame, 1024); assert.equal(first.frames, 256);
  assert.equal(first.warmup_discarded_frames, 128);
  assert.ok(!messages.some(m => m.type === 'gap'));
  context.currentFrame = 1536; recorder.process(input, []);
  assert.ok(messages.some(m => m.type === 'gap' && m.reason === 'frame_discontinuity'));
});
