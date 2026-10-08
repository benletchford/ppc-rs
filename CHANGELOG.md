# Changelog

## [0.7.1](https://github.com/benletchford/ppc-rs/compare/ppc-v0.7.0...ppc-v0.7.1) (2026-10-08)


### Bug Fixes

* **fpu:** round fsqrt, fsqrts and frsqrte square roots correctly in every mode ([ce4c218](https://github.com/benletchford/ppc-rs/commit/ce4c2181da48044f1058a261403f43cd25ceba20))
* **ppc:** defer traces until native callbacks return ([#78](https://github.com/benletchford/ppc-rs/issues/78)) ([ad4b726](https://github.com/benletchford/ppc-rs/commit/ad4b726c66744c82a4416b61862090f9ed44772e))


### Performance Improvements

* **fpu:** evaluate frsqrte natively with exact residual proofs ([4859b71](https://github.com/benletchford/ppc-rs/commit/4859b71028139b1a24d6c4b0204cfecc030d7b02))

## [0.7.0](https://github.com/benletchford/ppc-rs/compare/ppc-v0.6.6...ppc-v0.7.0) (2026-10-06)


### Features

* **memory:** expose guarded contiguous read spans ([#72](https://github.com/benletchford/ppc-rs/issues/72)) ([737d4fc](https://github.com/benletchford/ppc-rs/commit/737d4fc53644ce1422f41e5144747f55e1d8a8eb))
* **ppc:** allow guarded traces before cached blocks ([#74](https://github.com/benletchford/ppc-rs/issues/74)) ([8404b76](https://github.com/benletchford/ppc-rs/commit/8404b769124210701b12b132dbf6d08085013b81))
* **ppc:** compile guarded read-only traces to WebAssembly ([#76](https://github.com/benletchford/ppc-rs/issues/76)) ([028cd7b](https://github.com/benletchford/ppc-rs/commit/028cd7bac2837d71215cc7c5d6e3b5e450d774da))


### Performance Improvements

* avoid repeated PPC cached-block stop checks ([cae8f50](https://github.com/benletchford/ppc-rs/commit/cae8f5086616848f35328981e5ba779426a1a176))
* **ppc:** fuse cached word copy pairs ([#61](https://github.com/benletchford/ppc-rs/issues/61)) ([cdfc991](https://github.com/benletchford/ppc-rs/commit/cdfc991b759bd4d6497bb09c9114db3bcc8cb398))
* **ppc:** remove redundant cached block PC checks ([#62](https://github.com/benletchford/ppc-rs/issues/62)) ([929068d](https://github.com/benletchford/ppc-rs/commit/929068d07261e517ad8d7050c90a97a0baefb8a6))
* **ppc:** reuse global instruction mapping token on cached block hits ([#69](https://github.com/benletchford/ppc-rs/issues/69)) ([763be03](https://github.com/benletchford/ppc-rs/commit/763be033a418f90d441d8559fba350b415482cfb))
* skip redundant cached block end checks ([64e9428](https://github.com/benletchford/ppc-rs/commit/64e94283b1499720cfde8d89493d4eca5f7cc990))

## [0.6.6](https://github.com/benletchford/ppc-rs/compare/ppc-v0.6.5...ppc-v0.6.6) (2026-09-26)


### Performance Improvements

* **memory:** read words within visible overlay spans ([#48](https://github.com/benletchford/ppc-rs/issues/48)) ([cd147ae](https://github.com/benletchford/ppc-rs/commit/cd147aecf75279529448e433b54ec26fa96dcfff))
* **memory:** restore direct nonoverlap word reads ([#51](https://github.com/benletchford/ppc-rs/issues/51)) ([7227769](https://github.com/benletchford/ppc-rs/commit/7227769e4c5f0a7be23c54aa2587c19dba28b06c))
* skip unused visibility scans for writable instruction fetches ([1c77db9](https://github.com/benletchford/ppc-rs/commit/1c77db91b4bacfc796a970f3581f39eabebba2b1))

## [0.6.5](https://github.com/benletchford/ppc-rs/compare/ppc-v0.6.4...ppc-v0.6.5) (2026-09-25)


### Performance Improvements

* copy visible PPC memory spans in bulk ([#45](https://github.com/benletchford/ppc-rs/issues/45)) ([d5e477f](https://github.com/benletchford/ppc-rs/commit/d5e477f99ee76651589f72f638e69be9639b2cb6))

## [0.6.4](https://github.com/benletchford/ppc-rs/compare/ppc-v0.6.3...ppc-v0.6.4) (2026-09-21)


### Bug Fixes

* **cache:** revalidate blocks after guest stores ([#42](https://github.com/benletchford/ppc-rs/issues/42)) ([26b584c](https://github.com/benletchford/ppc-rs/commit/26b584c5f91f23b7d3a63dc112a95441c4064f62))

## [0.6.3](https://github.com/benletchford/ppc-rs/compare/ppc-v0.6.2...ppc-v0.6.3) (2026-09-20)


### Performance Improvements

* cache decoded instructions in immutable blocks ([#36](https://github.com/benletchford/ppc-rs/issues/36)) ([f636963](https://github.com/benletchford/ppc-rs/commit/f6369639bcc3b46a8afcab77984e7e27ebafe477))
* classify cached fast instructions ([#39](https://github.com/benletchford/ppc-rs/issues/39)) ([19ba796](https://github.com/benletchford/ppc-rs/commit/19ba796274daf1d151f696eaea9731a61c6e7a8a))

## [0.6.2](https://github.com/benletchford/ppc-rs/compare/ppc-v0.6.1...ppc-v0.6.2) (2026-09-16)


### Performance Improvements

* **exec:** accelerate section memory access and expand interpreter fast-path opcodes ([#33](https://github.com/benletchford/ppc-rs/issues/33)) ([8e5af9a](https://github.com/benletchford/ppc-rs/commit/8e5af9a63155de8dc92a7c826245075dc114c5fa))

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
