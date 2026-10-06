import { readFile } from 'node:fs/promises';

const [path, kind, input, limit] = process.argv.slice(2);
const bytes = await readFile(path);
if (!WebAssembly.validate(bytes)) throw Error('invalid generated WebAssembly');
const memory = new WebAssembly.Memory({ initial: 1 });
const data = new DataView(memory.buffer);
const state = 1024;
const source = 2048;
const entry = kind === 'read' ? 0x1000 : 0x3000;
data.setUint32(state + 128, entry, true);
data.setUint32(state + 3 * 4, 7, true);
data.setUint32(state + 4 * 4, 0x2000, true);
data.setUint32(state + 132, 19, true);
data.setBigUint64(state + 136, 42n, true);
data.setUint32(state + 144, 0x12345678, true);
data.setUint32(state + 148, 0x80000000, true);
data.setUint32(source, Number(input), false);
const instance = (await WebAssembly.instantiate(bytes, { env: { memory } })).instance;
const cycles = instance.exports.run(state, Number(limit), 0x2000, source, 4);
const result = [
  cycles,
  data.getUint32(state + 128, true),
  data.getUint32(state + 3 * 4, true),
  data.getUint32(state + 132, true),
  data.getBigUint64(state + 136, true),
  data.getUint32(state + 144, true),
  data.getUint32(state + 148, true),
];
if (kind === 'read') {
  data.setUint32(state + 128, entry, true);
  const initial = new Uint8Array(memory.buffer, state, 152).slice();
  if (instance.exports.run(state, 2, 0x2000, source, 4) !== 0
      || !initial.every((value, index) => value === new Uint8Array(memory.buffer, state, 152)[index])) {
    throw Error('short cycle budget changed architectural state');
  }
  if (instance.exports.run(state, 100, 0x2000, source, 3) !== 0
      || !initial.every((value, index) => value === new Uint8Array(memory.buffer, state, 152)[index])) {
    throw Error('short data span changed architectural state');
  }
  data.setUint32(state + 4 * 4, 0x2001, true);
  const unaligned = new Uint8Array(memory.buffer, state, 152).slice();
  if (instance.exports.run(state, 100, 0x2000, source, 4) !== 0
      || !unaligned.every((value, index) => value === new Uint8Array(memory.buffer, state, 152)[index])) {
    throw Error('unaligned guest load changed architectural state');
  }
}
console.log(result.join(','));
