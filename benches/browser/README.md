# Cached-block benchmark in Chrome

This runs the same four instruction loops as `benches/cached_blocks.rs` as WebAssembly in headless Chrome. Install the `wasm32-unknown-unknown` Rust target and use Node.js with a global `WebSocket`, then run:

```sh
node benches/browser/run.mjs
```

Set `CHROME_BIN` if Chrome or Chromium is outside the usual macOS/Linux locations. `PPC_BROWSER_BENCH_CYCLES` and `PPC_BROWSER_BENCH_RUNS` control the measured cycles and repetitions (defaults: five million and five). The script builds the local `ppc` crate, serves the Wasm module on loopback, measures each case with `performance.now()`, verifies a stable nonzero checksum, and prints the median time. It requires no website or game archive.

Run the unchanged and candidate revisions on the same device and browser, in alternating order. These loops isolate the PPC interpreter and `PpcSectionMem`; they do not include the full guest address-space router, toolbox calls, presentation, or game behavior. A browser game workload is still the final performance gate.
