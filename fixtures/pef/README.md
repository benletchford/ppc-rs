These PEF inventory fixtures capture representative Classic Mac PowerPC
application fragments. Regenerate them after intentional `ppc-inspect` JSON
schema changes by providing local paths to the corresponding PEF files:

```sh
cargo run --quiet --bin ppc-inspect -- --no-path path/to/Nanosaur > fixtures/pef/nanosaur.json
cargo run --quiet --bin ppc-inspect -- --no-path path/to/Nanosaur-2MB > fixtures/pef/nanosaur-2mb.json
```
