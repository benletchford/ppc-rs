# Changelog

## [0.6.1](https://github.com/benletchford/ppc-rs/compare/ppc-v0.6.0...ppc-v0.6.1) (2026-09-08)


### Bug Fixes

* **ppc:** respect CFM import cycle budgets ([cf1d9ae](https://github.com/benletchford/ppc-rs/commit/cf1d9ae10bab1346c7864fb60072fc76eb137156))

## [0.6.0](https://github.com/benletchford/ppc-rs/compare/ppc-v0.5.0...ppc-v0.6.0) (2026-09-06)


### Features

* separate suspendable PPC execution contexts from engine state ([f76dcc4](https://github.com/benletchford/ppc-rs/commit/f76dcc42f2e0e36e7dffdadf8613f8cb536c098b))


### Bug Fixes

* distinguish immutable instruction mappings across memories ([3b1bb99](https://github.com/benletchford/ppc-rs/commit/3b1bb99d6c0d54142cb1972f59f19e41c597fdac))
* revalidate CFM import stubs through instruction fetches ([765e668](https://github.com/benletchford/ppc-rs/commit/765e6687b4ab532ad3621447a9512344888b23d4))

## [0.5.0](https://github.com/benletchford/ppc-rs/compare/ppc-v0.4.1...ppc-v0.5.0) (2026-08-29)


### Features

* **import:** continue from handler-arranged CPU state ([#20](https://github.com/benletchford/ppc-rs/issues/20)) ([4743cda](https://github.com/benletchford/ppc-rs/commit/4743cdafbef4f964ff52fc63550e4c64b9f46763))

## [0.4.1](https://github.com/benletchford/ppc-rs/compare/ppc-v0.4.0...ppc-v0.4.1) (2026-08-28)


### Bug Fixes

* enforce 32-bit PowerPC architectural semantics ([#17](https://github.com/benletchford/ppc-rs/issues/17)) ([681f83b](https://github.com/benletchford/ppc-rs/commit/681f83b9e92ab8643c96a9986b5254db05f79b8e))

## [0.4.0](https://github.com/benletchford/ppc-rs/compare/ppc-v0.3.1...ppc-v0.4.0) (2026-08-17)


### Features

* implement PowerPC time-base reads ([#14](https://github.com/benletchford/ppc-rs/issues/14)) ([f2b115f](https://github.com/benletchford/ppc-rs/commit/f2b115f477155ed4f81a7e6c8e01920f99773839))

## [0.3.1](https://github.com/benletchford/ppc-rs/compare/ppc-v0.3.0...ppc-v0.3.1) (2026-08-16)


### Performance Improvements

* cache immutable PowerPC basic blocks ([#8](https://github.com/benletchford/ppc-rs/issues/8)) ([9bb9bdb](https://github.com/benletchford/ppc-rs/commit/9bb9bdbbe8c0accedcfffb9444fa94c1cd8fd079))

## [0.3.0](https://github.com/benletchford/ppc-rs/compare/ppc-v0.2.0...ppc-v0.3.0) (2026-08-16)


### Features

* complete remaining 32-bit user instruction coverage ([#5](https://github.com/benletchford/ppc-rs/issues/5)) ([9ada49f](https://github.com/benletchford/ppc-rs/commit/9ada49f4f7318b536dce38707ca8b4fc7ba2cbc3))

## [0.2.0](https://github.com/benletchford/ppc-rs/releases/tag/ppc-v0.2.0) (2026-08-16)


### Features

* add standalone PowerPC interpreter ([c721e59](https://github.com/benletchford/ppc-rs/commit/c721e59d2f17e24ba358e098d3132b900e3905ad))
