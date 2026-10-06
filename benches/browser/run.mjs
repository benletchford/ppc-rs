#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const benchDir = dirname(fileURLToPath(import.meta.url));
const crateRoot = resolve(benchDir, '../..');
const targetDir = join(crateRoot, 'target/browser-bench');
const manifest = join(benchDir, 'Cargo.toml');
const wasmPath = join(targetDir, 'wasm32-unknown-unknown/release/ppc_browser_bench.wasm');
const cycles = Number(process.env.PPC_BROWSER_BENCH_CYCLES ?? 5_000_000);
const runs = Number(process.env.PPC_BROWSER_BENCH_RUNS ?? 5);
if (!Number.isSafeInteger(cycles) || cycles < 100_000 || cycles > 1_000_000_000
    || !Number.isSafeInteger(runs) || runs < 1 || runs > 30) {
  throw new Error('PPC_BROWSER_BENCH_CYCLES or PPC_BROWSER_BENCH_RUNS is invalid');
}
const chromePath = process.env.CHROME_BIN ?? [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].find(existsSync);
if (!chromePath || !existsSync(chromePath)) throw new Error('Set CHROME_BIN to a Chrome or Chromium executable');
if (typeof WebSocket !== 'function') throw new Error('Node.js with a global WebSocket is required');

await runCommand('cargo', [
  'build', '--release', '--locked', '--target', 'wasm32-unknown-unknown',
  '--manifest-path', manifest,
], {
  ...process.env,
  CARGO_TARGET_DIR: targetDir,
  RUSTFLAGS: `${process.env.RUSTFLAGS ?? ''} -C target-feature=+simd128`.trim(),
});
const wasm = await readFile(wasmPath);
const server = createServer((request, response) => {
  if (request.url === '/bench.wasm') {
    response.writeHead(200, { 'content-type': 'application/wasm', 'content-length': wasm.length });
    response.end(wasm);
  } else if (request.url === '/') {
    response.writeHead(200, { 'content-type': 'text/html' });
    response.end('<!doctype html><title>PPC browser benchmark</title>');
  } else {
    response.writeHead(404);
    response.end();
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/`;
const userDataDir = await mkdtemp(join(tmpdir(), 'ppc-browser-bench-'));
const debugPort = 10000 + Math.floor(Math.random() * 30000);
const chrome = spawn(chromePath, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${userDataDir}`,
  '--no-first-run', '--no-default-browser-check', 'about:blank',
], { stdio: 'ignore' });
try {
  const version = await waitForJson(`http://127.0.0.1:${debugPort}/json/version`);
  const browser = connect(version.webSocketDebuggerUrl);
  await browser.ready;
  const { targetId } = await browser.send('Target.createTarget', { url: 'about:blank' });
  browser.close();
  const targets = await waitForJson(`http://127.0.0.1:${debugPort}/json/list`);
  const target = targets.find(candidate => candidate.id === targetId);
  if (!target) throw new Error('Chrome did not expose the benchmark tab');
  const page = connect(target.webSocketDebuggerUrl);
  await page.ready;
  await page.send('Page.enable');
  await page.send('Runtime.enable');
  await page.send('Page.navigate', { url });
  for (let attempt = 0; attempt < 200; attempt++) {
    try {
      const state = await page.send('Runtime.evaluate', {
        expression: '({ url: location.href, ready: document.readyState })', returnByValue: true,
      });
      if (state.result.value.url === url && state.result.value.ready === 'complete') break;
    } catch { /* Navigation can replace the execution context. */ }
    if (attempt === 199) throw new Error('Benchmark tab did not finish loading');
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  const expression = `(async () => {
    const bytes = await (await fetch('/bench.wasm')).arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes);
    const names = ['arithmetic', 'conditional', 'load_store', 'copy_words'];
    const results = [];
    for (let kind = 0; kind < names.length; kind++) {
      instance.exports.run_bench(kind, ${Math.max(100_000, Math.floor(cycles / 5))});
      const times = [];
      let expected;
      for (let repeat = 0; repeat < ${runs}; repeat++) {
        const start = performance.now();
        const checksum = instance.exports.run_bench(kind, ${cycles}) >>> 0;
        const ms = performance.now() - start;
        if (!checksum || (expected !== undefined && checksum !== expected)) {
          throw new Error('Unexpected benchmark result for ' + names[kind] + ': ' + checksum);
        }
        expected = checksum;
        times.push(ms);
      }
      times.sort((a, b) => a - b);
      results.push({ name: names[kind], medianMs: times[Math.floor(times.length / 2)], checksum: expected });
    }
    return results;
  })()`;
  const evaluated = await page.send('Runtime.evaluate', {
    expression, returnByValue: true, awaitPromise: true, timeout: 120_000,
  });
  if (evaluated.exceptionDetails) throw new Error(JSON.stringify(evaluated.exceptionDetails));
  console.log(version.Browser);
  console.log('case\tcycles\tmedian_ms\tMcycles/s\tchecksum');
  for (const { name, medianMs, checksum } of evaluated.result.value) {
    console.log(`${name}\t${cycles}\t${medianMs.toFixed(1)}\t${(cycles / medianMs / 1000).toFixed(2)}\t${checksum}`);
  }
  page.close();
} finally {
  chrome.kill('SIGTERM');
  await new Promise(resolve => setTimeout(resolve, 250));
  await rm(userDataDir, { recursive: true, force: true });
  await new Promise(resolve => server.close(resolve));
}

function runCommand(command, args, env) {
  return new Promise((resolve, reject) => {
    const process = spawn(command, args, { stdio: 'inherit', env });
    process.on('error', reject);
    process.on('exit', code => code === 0 ? resolve() : reject(new Error(`${command} exited ${code}`)));
  });
}

async function waitForJson(url) {
  for (let attempt = 0; attempt < 200; attempt++) {
    try { return await (await fetch(url)).json(); }
    catch { await new Promise(resolve => setTimeout(resolve, 50)); }
  }
  throw new Error(`Chrome endpoint did not respond: ${url}`);
}

function connect(webSocketUrl) {
  const socket = new WebSocket(webSocketUrl);
  const pending = new Map();
  let nextId = 1;
  const ready = new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    const item = pending.get(message.id);
    if (!item) return;
    pending.delete(message.id);
    if (message.error) item.reject(new Error(JSON.stringify(message.error)));
    else item.resolve(message.result ?? {});
  });
  return {
    ready,
    close() { socket.close(); },
    send(method, params = {}) {
      const id = nextId++;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        socket.send(JSON.stringify({ id, method, params }));
      });
    },
  };
}
