PEF inventory fixtures generated from `systemless-games/nanosaur.sit`.

Regenerate after intentional `ppc-inspect` JSON schema changes:

```sh
tmpdir=$(mktemp -d /tmp/systemless-nanosaur-fixtures.XXXXXX)
unar -q -o "$tmpdir" systemless-games/nanosaur.sit
default_app=$(find "$tmpdir/nanosaur" -maxdepth 1 -type f -name 'Nanosaur*' ! -name '*2MB*' ! -name '*.pdf' -print -quit)
low_mem_app=$(find "$tmpdir/nanosaur" -maxdepth 1 -type f -name 'Nanosaur*2MB*' -print -quit)
cargo run --quiet --manifest-path ppc-rs/Cargo.toml --bin ppc-inspect -- --no-path "$default_app" > ppc-rs/fixtures/pef/nanosaur.json
cargo run --quiet --manifest-path ppc-rs/Cargo.toml --bin ppc-inspect -- --no-path "$low_mem_app" > ppc-rs/fixtures/pef/nanosaur-2mb.json
```
