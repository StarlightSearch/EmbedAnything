embed_anything v0.5.1 (/home/akshay/projects/EmbedAnything/rust)
├── anyhow v1.0.95
├── base64 v0.22.1
├── byteorder v1.5.0
├── candle-core v0.8.2 (https://github.com/huggingface/candle.git#efd0e682)
│   ├── byteorder v1.5.0
│   ├── gemm v0.17.1
│   │   ├── dyn-stack v0.10.0
│   │   │   ├── bytemuck v1.21.0
│   │   │   │   └── bytemuck_derive v1.8.1 (proc-macro)
│   │   │   │       ├── proc-macro2 v1.0.93
│   │   │   │       │   └── unicode-ident v1.0.14
│   │   │   │       ├── quote v1.0.38
│   │   │   │       │   └── proc-macro2 v1.0.93 (*)
│   │   │   │       └── syn v2.0.96
│   │   │   │           ├── proc-macro2 v1.0.93 (*)
│   │   │   │           ├── quote v1.0.38 (*)
│   │   │   │           └── unicode-ident v1.0.14
│   │   │   └── reborrow v0.5.5
│   │   ├── gemm-c32 v0.17.1
│   │   │   ├── dyn-stack v0.10.0 (*)
│   │   │   ├── gemm-common v0.17.1
│   │   │   │   ├── bytemuck v1.21.0 (*)
│   │   │   │   ├── dyn-stack v0.10.0 (*)
│   │   │   │   ├── half v2.4.1
│   │   │   │   │   ├── bytemuck v1.21.0 (*)
│   │   │   │   │   ├── cfg-if v1.0.0
│   │   │   │   │   ├── num-traits v0.2.19
│   │   │   │   │   │   └── libm v0.2.11
│   │   │   │   │   │   [build-dependencies]
│   │   │   │   │   │   └── autocfg v1.4.0
│   │   │   │   │   ├── rand v0.8.5
│   │   │   │   │   │   ├── libc v0.2.169
│   │   │   │   │   │   ├── rand_chacha v0.3.1
│   │   │   │   │   │   │   ├── ppv-lite86 v0.2.20
│   │   │   │   │   │   │   │   └── zerocopy v0.7.35
│   │   │   │   │   │   │   │       ├── byteorder v1.5.0
│   │   │   │   │   │   │   │       └── zerocopy-derive v0.7.35 (proc-macro)
│   │   │   │   │   │   │   │           ├── proc-macro2 v1.0.93 (*)
│   │   │   │   │   │   │   │           ├── quote v1.0.38 (*)
│   │   │   │   │   │   │   │           └── syn v2.0.96 (*)
│   │   │   │   │   │   │   └── rand_core v0.6.4
│   │   │   │   │   │   │       └── getrandom v0.2.15
│   │   │   │   │   │   │           ├── cfg-if v1.0.0
│   │   │   │   │   │   │           └── libc v0.2.169
│   │   │   │   │   │   └── rand_core v0.6.4 (*)
│   │   │   │   │   └── rand_distr v0.4.3
│   │   │   │   │       ├── num-traits v0.2.19 (*)
│   │   │   │   │       └── rand v0.8.5 (*)
│   │   │   │   ├── num-complex v0.4.6
│   │   │   │   │   ├── bytemuck v1.21.0 (*)
│   │   │   │   │   ├── num-traits v0.2.19 (*)
│   │   │   │   │   ├── rand v0.8.5 (*)
│   │   │   │   │   └── serde v1.0.217
│   │   │   │   │       └── serde_derive v1.0.217 (proc-macro)
│   │   │   │   │           ├── proc-macro2 v1.0.93 (*)
│   │   │   │   │           ├── quote v1.0.38 (*)
│   │   │   │   │           └── syn v2.0.96 (*)
│   │   │   │   ├── num-traits v0.2.19 (*)
│   │   │   │   ├── once_cell v1.20.2
│   │   │   │   ├── paste v1.0.15 (proc-macro)
│   │   │   │   ├── pulp v0.18.22
│   │   │   │   │   ├── bytemuck v1.21.0 (*)
│   │   │   │   │   ├── libm v0.2.11
│   │   │   │   │   ├── num-complex v0.4.6 (*)
│   │   │   │   │   └── reborrow v0.5.5
│   │   │   │   ├── raw-cpuid v10.7.0
│   │   │   │   │   └── bitflags v1.3.2
│   │   │   │   ├── rayon v1.10.0
│   │   │   │   │   ├── either v1.13.0
│   │   │   │   │   └── rayon-core v1.12.1
│   │   │   │   │       ├── crossbeam-deque v0.8.6
│   │   │   │   │       │   ├── crossbeam-epoch v0.9.18
│   │   │   │   │       │   │   └── crossbeam-utils v0.8.21
│   │   │   │   │       │   └── crossbeam-utils v0.8.21
│   │   │   │   │       └── crossbeam-utils v0.8.21
│   │   │   │   └── seq-macro v0.3.5 (proc-macro)
│   │   │   ├── num-complex v0.4.6 (*)
│   │   │   ├── num-traits v0.2.19 (*)
│   │   │   ├── paste v1.0.15 (proc-macro)
│   │   │   ├── raw-cpuid v10.7.0 (*)
│   │   │   └── seq-macro v0.3.5 (proc-macro)
│   │   ├── gemm-c64 v0.17.1
│   │   │   ├── dyn-stack v0.10.0 (*)
│   │   │   ├── gemm-common v0.17.1 (*)
│   │   │   ├── num-complex v0.4.6 (*)
│   │   │   ├── num-traits v0.2.19 (*)
│   │   │   ├── paste v1.0.15 (proc-macro)
│   │   │   ├── raw-cpuid v10.7.0 (*)
│   │   │   └── seq-macro v0.3.5 (proc-macro)
│   │   ├── gemm-common v0.17.1 (*)
│   │   ├── gemm-f16 v0.17.1
│   │   │   ├── dyn-stack v0.10.0 (*)
│   │   │   ├── gemm-common v0.17.1 (*)
│   │   │   ├── gemm-f32 v0.17.1
│   │   │   │   ├── dyn-stack v0.10.0 (*)
│   │   │   │   ├── gemm-common v0.17.1 (*)
│   │   │   │   ├── num-complex v0.4.6 (*)
│   │   │   │   ├── num-traits v0.2.19 (*)
│   │   │   │   ├── paste v1.0.15 (proc-macro)
│   │   │   │   ├── raw-cpuid v10.7.0 (*)
│   │   │   │   └── seq-macro v0.3.5 (proc-macro)
│   │   │   ├── half v2.4.1 (*)
│   │   │   ├── num-complex v0.4.6 (*)
│   │   │   ├── num-traits v0.2.19 (*)
│   │   │   ├── paste v1.0.15 (proc-macro)
│   │   │   ├── raw-cpuid v10.7.0 (*)
│   │   │   ├── rayon v1.10.0 (*)
│   │   │   └── seq-macro v0.3.5 (proc-macro)
│   │   ├── gemm-f32 v0.17.1 (*)
│   │   ├── gemm-f64 v0.17.1
│   │   │   ├── dyn-stack v0.10.0 (*)
│   │   │   ├── gemm-common v0.17.1 (*)
│   │   │   ├── num-complex v0.4.6 (*)
│   │   │   ├── num-traits v0.2.19 (*)
│   │   │   ├── paste v1.0.15 (proc-macro)
│   │   │   ├── raw-cpuid v10.7.0 (*)
│   │   │   └── seq-macro v0.3.5 (proc-macro)
│   │   ├── num-complex v0.4.6 (*)
│   │   ├── num-traits v0.2.19 (*)
│   │   ├── paste v1.0.15 (proc-macro)
│   │   ├── raw-cpuid v10.7.0 (*)
│   │   └── seq-macro v0.3.5 (proc-macro)
│   ├── half v2.4.1 (*)
│   ├── memmap2 v0.9.5
│   │   ├── libc v0.2.169
│   │   └── stable_deref_trait v1.2.0
│   ├── num-traits v0.2.19 (*)
│   ├── num_cpus v1.16.0
│   │   └── libc v0.2.169
│   ├── rand v0.8.5 (*)
│   ├── rand_distr v0.4.3 (*)
│   ├── rayon v1.10.0 (*)
│   ├── safetensors v0.4.5
│   │   ├── serde v1.0.217 (*)
│   │   └── serde_json v1.0.135
│   │       ├── itoa v1.0.14
│   │       ├── memchr v2.7.4
│   │       ├── ryu v1.0.18
│   │       └── serde v1.0.217 (*)
│   ├── thiserror v1.0.69
│   │   └── thiserror-impl v1.0.69 (proc-macro)
│   │       ├── proc-macro2 v1.0.93 (*)
│   │       ├── quote v1.0.38 (*)
│   │       └── syn v2.0.96 (*)
│   ├── ug v0.0.2
│   │   ├── half v2.4.1 (*)
│   │   ├── num v0.4.3
│   │   │   ├── num-bigint v0.4.6
│   │   │   │   ├── num-integer v0.1.46
│   │   │   │   │   └── num-traits v0.2.19 (*)
│   │   │   │   └── num-traits v0.2.19 (*)
│   │   │   ├── num-complex v0.4.6 (*)
│   │   │   ├── num-integer v0.1.46 (*)
│   │   │   ├── num-iter v0.1.45
│   │   │   │   ├── num-integer v0.1.46 (*)
│   │   │   │   └── num-traits v0.2.19 (*)
│   │   │   │   [build-dependencies]
│   │   │   │   └── autocfg v1.4.0
│   │   │   ├── num-rational v0.4.2
│   │   │   │   ├── num-bigint v0.4.6 (*)
│   │   │   │   ├── num-integer v0.1.46 (*)
│   │   │   │   └── num-traits v0.2.19 (*)
│   │   │   └── num-traits v0.2.19 (*)
│   │   ├── serde v1.0.217 (*)
│   │   ├── serde_json v1.0.135 (*)
│   │   └── thiserror v1.0.69 (*)
│   ├── yoke v0.7.5
│   │   ├── stable_deref_trait v1.2.0
│   │   ├── yoke-derive v0.7.5 (proc-macro)
│   │   │   ├── proc-macro2 v1.0.93 (*)
│   │   │   ├── quote v1.0.38 (*)
│   │   │   ├── syn v2.0.96 (*)
│   │   │   └── synstructure v0.13.1
│   │   │       ├── proc-macro2 v1.0.93 (*)
│   │   │       ├── quote v1.0.38 (*)
│   │   │       └── syn v2.0.96 (*)
│   │   └── zerofrom v0.1.5
│   │       └── zerofrom-derive v0.1.5 (proc-macro)
│   │           ├── proc-macro2 v1.0.93 (*)
│   │           ├── quote v1.0.38 (*)
│   │           ├── syn v2.0.96 (*)
│   │           └── synstructure v0.13.1 (*)
│   └── zip v1.1.4
│       ├── crc32fast v1.4.2
│       │   └── cfg-if v1.0.0
│       ├── displaydoc v0.2.5 (proc-macro)
│       │   ├── proc-macro2 v1.0.93 (*)
│       │   ├── quote v1.0.38 (*)
│       │   └── syn v2.0.96 (*)
│       ├── flate2 v1.0.35
│       │   ├── crc32fast v1.4.2 (*)
│       │   └── miniz_oxide v0.8.3
│       │       ├── adler2 v2.0.0
│       │       └── simd-adler32 v0.3.7
│       ├── indexmap v2.7.0
│       │   ├── equivalent v1.0.1
│       │   └── hashbrown v0.15.2
│       ├── num_enum v0.7.3
│       │   └── num_enum_derive v0.7.3 (proc-macro)
│       │       ├── proc-macro-crate v3.2.0
│       │       │   └── toml_edit v0.22.22
│       │       │       ├── indexmap v2.7.0 (*)
│       │       │       ├── toml_datetime v0.6.8
│       │       │       └── winnow v0.6.24
│       │       ├── proc-macro2 v1.0.93 (*)
│       │       ├── quote v1.0.38 (*)
│       │       └── syn v2.0.96 (*)
│       └── thiserror v1.0.69 (*)
├── candle-nn v0.8.2 (https://github.com/huggingface/candle.git#efd0e682)
│   ├── candle-core v0.8.2 (https://github.com/huggingface/candle.git#efd0e682) (*)
│   ├── half v2.4.1 (*)
│   ├── num-traits v0.2.19 (*)
│   ├── rayon v1.10.0 (*)
│   ├── safetensors v0.4.5 (*)
│   ├── serde v1.0.217 (*)
│   └── thiserror v1.0.69 (*)
├── candle-transformers v0.8.2 (https://github.com/huggingface/candle.git#efd0e682)
│   ├── byteorder v1.5.0
│   ├── candle-core v0.8.2 (https://github.com/huggingface/candle.git#efd0e682) (*)
│   ├── candle-nn v0.8.2 (https://github.com/huggingface/candle.git#efd0e682) (*)
│   ├── fancy-regex v0.13.0
│   │   ├── bit-set v0.5.3
│   │   │   └── bit-vec v0.6.3
│   │   ├── regex-automata v0.4.9
│   │   │   ├── aho-corasick v1.1.3
│   │   │   │   └── memchr v2.7.4
│   │   │   ├── memchr v2.7.4
│   │   │   └── regex-syntax v0.8.5
│   │   └── regex-syntax v0.8.5
│   ├── num-traits v0.2.19 (*)
│   ├── rand v0.8.5 (*)
│   ├── rayon v1.10.0 (*)
│   ├── serde v1.0.217 (*)
│   ├── serde_json v1.0.135 (*)
│   ├── serde_plain v1.0.2
│   │   └── serde v1.0.217 (*)
│   └── tracing v0.1.41
│       ├── pin-project-lite v0.2.16
│       ├── tracing-attributes v0.1.28 (proc-macro)
│       │   ├── proc-macro2 v1.0.93 (*)
│       │   ├── quote v1.0.38 (*)
│       │   └── syn v2.0.96 (*)
│       └── tracing-core v0.1.33
│           └── once_cell v1.20.2
├── chrono v0.4.39
│   ├── iana-time-zone v0.1.61
│   └── num-traits v0.2.19 (*)
├── docx-parser v0.1.1
│   ├── base64 v0.22.1
│   ├── clap v4.5.26
│   │   ├── clap_builder v4.5.26
│   │   │   ├── anstream v0.6.18
│   │   │   │   ├── anstyle v1.0.10
│   │   │   │   ├── anstyle-parse v0.2.6
│   │   │   │   │   └── utf8parse v0.2.2
│   │   │   │   ├── anstyle-query v1.1.2
│   │   │   │   ├── colorchoice v1.0.3
│   │   │   │   ├── is_terminal_polyfill v1.70.1
│   │   │   │   └── utf8parse v0.2.2
│   │   │   ├── anstyle v1.0.10
│   │   │   ├── clap_lex v0.7.4
│   │   │   └── strsim v0.11.1
│   │   └── clap_derive v4.5.24 (proc-macro)
│   │       ├── heck v0.5.0
│   │       ├── proc-macro2 v1.0.93 (*)
│   │       ├── quote v1.0.38 (*)
│   │       └── syn v2.0.96 (*)
│   ├── docx-rust v0.1.8
│   │   ├── derive_more v0.99.18 (proc-macro)
│   │   │   ├── convert_case v0.4.0
│   │   │   ├── proc-macro2 v1.0.93 (*)
│   │   │   ├── quote v1.0.38 (*)
│   │   │   └── syn v2.0.96 (*)
│   │   │   [build-dependencies]
│   │   │   └── rustc_version v0.4.1
│   │   │       └── semver v1.0.24
│   │   ├── hard-xml v1.36.0
│   │   │   ├── hard-xml-derive v1.36.0 (proc-macro)
│   │   │   │   ├── bitflags v2.7.0
│   │   │   │   ├── proc-macro2 v1.0.93 (*)
│   │   │   │   ├── quote v1.0.38 (*)
│   │   │   │   └── syn v1.0.109
│   │   │   │       ├── proc-macro2 v1.0.93 (*)
│   │   │   │       ├── quote v1.0.38 (*)
│   │   │   │       └── unicode-ident v1.0.14
│   │   │   ├── jetscii v0.5.3
│   │   │   ├── lazy_static v1.5.0
│   │   │   ├── memchr v2.7.4
│   │   │   └── xmlparser v0.13.6
│   │   ├── log v0.4.25
│   │   └── zip v1.1.4 (*)
│   ├── serde v1.0.217 (*)
│   └── serde_json v1.0.135 (*)
├── docx-rust v0.1.8 (*)
├── futures v0.3.31
│   ├── futures-channel v0.3.31
│   │   ├── futures-core v0.3.31
│   │   └── futures-sink v0.3.31
│   ├── futures-core v0.3.31
│   ├── futures-executor v0.3.31
│   │   ├── futures-core v0.3.31
│   │   ├── futures-task v0.3.31
│   │   └── futures-util v0.3.31
│   │       ├── futures-channel v0.3.31 (*)
│   │       ├── futures-core v0.3.31
│   │       ├── futures-io v0.3.31
│   │       ├── futures-macro v0.3.31 (proc-macro)
│   │       │   ├── proc-macro2 v1.0.93 (*)
│   │       │   ├── quote v1.0.38 (*)
│   │       │   └── syn v2.0.96 (*)
│   │       ├── futures-sink v0.3.31
│   │       ├── futures-task v0.3.31
│   │       ├── memchr v2.7.4
│   │       ├── pin-project-lite v0.2.16
│   │       ├── pin-utils v0.1.0
│   │       └── slab v0.4.9
│   │           [build-dependencies]
│   │           └── autocfg v1.4.0
│   ├── futures-io v0.3.31
│   ├── futures-sink v0.3.31
│   ├── futures-task v0.3.31
│   └── futures-util v0.3.31 (*)
├── half v2.4.1 (*)
├── hf-hub v0.3.2
│   ├── dirs v5.0.1
│   │   └── dirs-sys v0.4.1
│   │       ├── libc v0.2.169
│   │       └── option-ext v0.2.0
│   ├── indicatif v0.17.9
│   │   ├── console v0.15.10
│   │   │   ├── libc v0.2.169
│   │   │   ├── once_cell v1.20.2
│   │   │   └── unicode-width v0.2.0
│   │   ├── number_prefix v0.4.0
│   │   ├── portable-atomic v1.10.0
│   │   └── unicode-width v0.2.0
│   ├── log v0.4.25
│   ├── native-tls v0.2.12
│   │   ├── log v0.4.25
│   │   ├── openssl v0.10.68
│   │   │   ├── bitflags v2.7.0
│   │   │   ├── cfg-if v1.0.0
│   │   │   ├── foreign-types v0.3.2
│   │   │   │   └── foreign-types-shared v0.1.1
│   │   │   ├── libc v0.2.169
│   │   │   ├── once_cell v1.20.2
│   │   │   ├── openssl-macros v0.1.1 (proc-macro)
│   │   │   │   ├── proc-macro2 v1.0.93 (*)
│   │   │   │   ├── quote v1.0.38 (*)
│   │   │   │   └── syn v2.0.96 (*)
│   │   │   └── openssl-sys v0.9.104
│   │   │       └── libc v0.2.169
│   │   │       [build-dependencies]
│   │   │       ├── cc v1.2.9
│   │   │       │   └── shlex v1.3.0
│   │   │       ├── pkg-config v0.3.31
│   │   │       └── vcpkg v0.2.15
│   │   ├── openssl-probe v0.1.5
│   │   └── openssl-sys v0.9.104 (*)
│   ├── rand v0.8.5 (*)
│   ├── serde v1.0.217 (*)
│   ├── serde_json v1.0.135 (*)
│   ├── thiserror v1.0.69 (*)
│   └── ureq v2.12.1
│       ├── base64 v0.22.1
│       ├── flate2 v1.0.35 (*)
│       ├── log v0.4.25
│       ├── native-tls v0.2.12 (*)
│       ├── once_cell v1.20.2
│       ├── rustls v0.23.21
│       │   ├── log v0.4.25
│       │   ├── once_cell v1.20.2
│       │   ├── ring v0.17.8
│       │   │   ├── cfg-if v1.0.0
│       │   │   ├── getrandom v0.2.15 (*)
│       │   │   ├── spin v0.9.8
│       │   │   └── untrusted v0.9.0
│       │   │   [build-dependencies]
│       │   │   └── cc v1.2.9 (*)
│       │   ├── rustls-pki-types v1.10.1
│       │   ├── rustls-webpki v0.102.8
│       │   │   ├── ring v0.17.8 (*)
│       │   │   ├── rustls-pki-types v1.10.1
│       │   │   └── untrusted v0.9.0
│       │   ├── subtle v2.6.1
│       │   └── zeroize v1.8.1
│       ├── rustls-pki-types v1.10.1
│       ├── serde v1.0.217 (*)
│       ├── serde_json v1.0.135 (*)
│       ├── url v2.5.4
│       │   ├── form_urlencoded v1.2.1
│       │   │   └── percent-encoding v2.3.1
│       │   ├── idna v1.0.3
│       │   │   ├── idna_adapter v1.2.0
│       │   │   │   ├── icu_normalizer v1.5.0
│       │   │   │   │   ├── displaydoc v0.2.5 (proc-macro) (*)
│       │   │   │   │   ├── icu_collections v1.5.0
│       │   │   │   │   │   ├── displaydoc v0.2.5 (proc-macro) (*)
│       │   │   │   │   │   ├── yoke v0.7.5 (*)
│       │   │   │   │   │   ├── zerofrom v0.1.5 (*)
│       │   │   │   │   │   └── zerovec v0.10.4
│       │   │   │   │   │       ├── yoke v0.7.5 (*)
│       │   │   │   │   │       ├── zerofrom v0.1.5 (*)
│       │   │   │   │   │       └── zerovec-derive v0.10.3 (proc-macro)
│       │   │   │   │   │           ├── proc-macro2 v1.0.93 (*)
│       │   │   │   │   │           ├── quote v1.0.38 (*)
│       │   │   │   │   │           └── syn v2.0.96 (*)
│       │   │   │   │   ├── icu_normalizer_data v1.5.0
│       │   │   │   │   ├── icu_properties v1.5.1
│       │   │   │   │   │   ├── displaydoc v0.2.5 (proc-macro) (*)
│       │   │   │   │   │   ├── icu_collections v1.5.0 (*)
│       │   │   │   │   │   ├── icu_locid_transform v1.5.0
│       │   │   │   │   │   │   ├── displaydoc v0.2.5 (proc-macro) (*)
│       │   │   │   │   │   │   ├── icu_locid v1.5.0
│       │   │   │   │   │   │   │   ├── displaydoc v0.2.5 (proc-macro) (*)
│       │   │   │   │   │   │   │   ├── litemap v0.7.4
│       │   │   │   │   │   │   │   ├── tinystr v0.7.6
│       │   │   │   │   │   │   │   │   ├── displaydoc v0.2.5 (proc-macro) (*)
│       │   │   │   │   │   │   │   │   └── zerovec v0.10.4 (*)
│       │   │   │   │   │   │   │   ├── writeable v0.5.5
│       │   │   │   │   │   │   │   └── zerovec v0.10.4 (*)
│       │   │   │   │   │   │   ├── icu_locid_transform_data v1.5.0
│       │   │   │   │   │   │   ├── icu_provider v1.5.0
│       │   │   │   │   │   │   │   ├── displaydoc v0.2.5 (proc-macro) (*)
│       │   │   │   │   │   │   │   ├── icu_locid v1.5.0 (*)
│       │   │   │   │   │   │   │   ├── icu_provider_macros v1.5.0 (proc-macro)
│       │   │   │   │   │   │   │   │   ├── proc-macro2 v1.0.93 (*)
│       │   │   │   │   │   │   │   │   ├── quote v1.0.38 (*)
│       │   │   │   │   │   │   │   │   └── syn v2.0.96 (*)
│       │   │   │   │   │   │   │   ├── stable_deref_trait v1.2.0
│       │   │   │   │   │   │   │   ├── tinystr v0.7.6 (*)
│       │   │   │   │   │   │   │   ├── writeable v0.5.5
│       │   │   │   │   │   │   │   ├── yoke v0.7.5 (*)
│       │   │   │   │   │   │   │   ├── zerofrom v0.1.5 (*)
│       │   │   │   │   │   │   │   └── zerovec v0.10.4 (*)
│       │   │   │   │   │   │   ├── tinystr v0.7.6 (*)
│       │   │   │   │   │   │   └── zerovec v0.10.4 (*)
│       │   │   │   │   │   ├── icu_properties_data v1.5.0
│       │   │   │   │   │   ├── icu_provider v1.5.0 (*)
│       │   │   │   │   │   ├── tinystr v0.7.6 (*)
│       │   │   │   │   │   └── zerovec v0.10.4 (*)
│       │   │   │   │   ├── icu_provider v1.5.0 (*)
│       │   │   │   │   ├── smallvec v1.13.2
│       │   │   │   │   ├── utf16_iter v1.0.5
│       │   │   │   │   ├── utf8_iter v1.0.4
│       │   │   │   │   ├── write16 v1.0.0
│       │   │   │   │   └── zerovec v0.10.4 (*)
│       │   │   │   └── icu_properties v1.5.1 (*)
│       │   │   ├── smallvec v1.13.2
│       │   │   └── utf8_iter v1.0.4
│       │   └── percent-encoding v2.3.1
│       └── webpki-roots v0.26.7
│           └── rustls-pki-types v1.10.1
├── image v0.25.5
│   ├── bytemuck v1.21.0 (*)
│   ├── byteorder-lite v0.1.0
│   ├── color_quant v1.1.0
│   ├── exr v1.73.0
│   │   ├── bit_field v0.10.2
│   │   ├── half v2.4.1 (*)
│   │   ├── lebe v0.5.2
│   │   ├── miniz_oxide v0.8.3 (*)
│   │   ├── rayon-core v1.12.1 (*)
│   │   ├── smallvec v1.13.2
│   │   └── zune-inflate v0.2.54
│   │       └── simd-adler32 v0.3.7
│   ├── gif v0.13.1
│   │   ├── color_quant v1.1.0
│   │   └── weezl v0.1.8
│   ├── image-webp v0.2.1
│   │   ├── byteorder-lite v0.1.0
│   │   └── quick-error v2.0.1
│   ├── num-traits v0.2.19 (*)
│   ├── png v0.17.16
│   │   ├── bitflags v1.3.2
│   │   ├── crc32fast v1.4.2 (*)
│   │   ├── fdeflate v0.3.7
│   │   │   └── simd-adler32 v0.3.7
│   │   ├── flate2 v1.0.35 (*)
│   │   └── miniz_oxide v0.8.3 (*)
│   ├── qoi v0.4.1
│   │   └── bytemuck v1.21.0 (*)
│   ├── ravif v0.11.11
│   │   ├── avif-serialize v0.8.2
│   │   │   └── arrayvec v0.7.6
│   │   ├── imgref v1.11.0
│   │   ├── loop9 v0.1.5
│   │   │   └── imgref v1.11.0
│   │   ├── quick-error v2.0.1
│   │   ├── rav1e v0.7.1
│   │   │   ├── arg_enum_proc_macro v0.3.4 (proc-macro)
│   │   │   │   ├── proc-macro2 v1.0.93 (*)
│   │   │   │   ├── quote v1.0.38 (*)
│   │   │   │   └── syn v2.0.96 (*)
│   │   │   ├── arrayvec v0.7.6
│   │   │   ├── av1-grain v0.2.3
│   │   │   │   ├── anyhow v1.0.95
│   │   │   │   ├── arrayvec v0.7.6
│   │   │   │   ├── log v0.4.25
│   │   │   │   ├── nom v7.1.3
│   │   │   │   │   ├── memchr v2.7.4
│   │   │   │   │   └── minimal-lexical v0.2.1
│   │   │   │   ├── num-rational v0.4.2 (*)
│   │   │   │   └── v_frame v0.3.8
│   │   │   │       ├── aligned-vec v0.5.0
│   │   │   │       └── num-traits v0.2.19 (*)
│   │   │   ├── bitstream-io v2.6.0
│   │   │   ├── cfg-if v1.0.0
│   │   │   ├── itertools v0.12.1
│   │   │   │   └── either v1.13.0
│   │   │   ├── libc v0.2.169
│   │   │   ├── log v0.4.25
│   │   │   ├── maybe-rayon v0.1.1
│   │   │   │   ├── cfg-if v1.0.0
│   │   │   │   └── rayon v1.10.0 (*)
│   │   │   ├── new_debug_unreachable v1.0.6
│   │   │   ├── noop_proc_macro v0.3.0 (proc-macro)
│   │   │   ├── num-derive v0.4.2 (proc-macro)
│   │   │   │   ├── proc-macro2 v1.0.93 (*)
│   │   │   │   ├── quote v1.0.38 (*)
│   │   │   │   └── syn v2.0.96 (*)
│   │   │   ├── num-traits v0.2.19 (*)
│   │   │   ├── once_cell v1.20.2
│   │   │   ├── paste v1.0.15 (proc-macro)
│   │   │   ├── profiling v1.0.16
│   │   │   │   └── profiling-procmacros v1.0.16 (proc-macro)
│   │   │   │       ├── quote v1.0.38 (*)
│   │   │   │       └── syn v2.0.96 (*)
│   │   │   ├── simd_helpers v0.1.0 (proc-macro)
│   │   │   │   └── quote v1.0.38 (*)
│   │   │   ├── thiserror v1.0.69 (*)
│   │   │   └── v_frame v0.3.8 (*)
│   │   │   [build-dependencies]
│   │   │   └── built v0.7.5
│   │   ├── rayon v1.10.0 (*)
│   │   └── rgb v0.8.50
│   ├── rayon v1.10.0 (*)
│   ├── rgb v0.8.50
│   ├── tiff v0.9.1
│   │   ├── flate2 v1.0.35 (*)
│   │   ├── jpeg-decoder v0.3.1
│   │   └── weezl v0.1.8
│   ├── zune-core v0.4.12
│   └── zune-jpeg v0.4.14
│       └── zune-core v0.4.12
├── indicatif v0.17.9 (*)
├── itertools v0.13.0
│   └── either v1.13.0
├── markdown_to_text v1.0.0
│   └── pulldown-cmark v0.7.2
│       ├── bitflags v1.3.2
│       ├── getopts v0.2.21
│       │   └── unicode-width v0.1.14
│       ├── memchr v2.7.4
│       └── unicase v2.8.1
├── ndarray v0.16.1
│   ├── matrixmultiply v0.3.9
│   │   └── rawpointer v0.2.1
│   │   [build-dependencies]
│   │   └── autocfg v1.4.0
│   ├── num-complex v0.4.6 (*)
│   ├── num-integer v0.1.46 (*)
│   ├── num-traits v0.2.19 (*)
│   └── rawpointer v0.2.1
├── ndarray-linalg v0.16.0
│   ├── cauchy v0.4.0
│   │   ├── num-complex v0.4.6 (*)
│   │   ├── num-traits v0.2.19 (*)
│   │   ├── rand v0.8.5 (*)
│   │   └── serde v1.0.217 (*)
│   ├── katexit v0.1.4 (proc-macro)
│   │   ├── proc-macro2 v1.0.93 (*)
│   │   ├── quote v1.0.38 (*)
│   │   └── syn v1.0.109 (*)
│   ├── lax v0.16.0
│   │   ├── cauchy v0.4.0 (*)
│   │   ├── katexit v0.1.4 (proc-macro) (*)
│   │   ├── lapack-sys v0.14.0
│   │   │   └── libc v0.2.169
│   │   ├── num-traits v0.2.19 (*)
│   │   └── thiserror v1.0.69 (*)
│   ├── ndarray v0.15.6
│   │   ├── approx v0.4.0
│   │   │   └── num-traits v0.2.19 (*)
│   │   ├── cblas-sys v0.1.4
│   │   │   └── libc v0.2.169
│   │   ├── libc v0.2.169
│   │   ├── matrixmultiply v0.3.9 (*)
│   │   ├── num-complex v0.4.6 (*)
│   │   ├── num-integer v0.1.46 (*)
│   │   ├── num-traits v0.2.19 (*)
│   │   └── rawpointer v0.2.1
│   ├── num-complex v0.4.6 (*)
│   ├── num-traits v0.2.19 (*)
│   ├── rand v0.8.5 (*)
│   └── thiserror v1.0.69 (*)
├── ort v2.0.0-rc.9
│   ├── half v2.4.1 (*)
│   ├── libloading v0.8.6
│   │   └── cfg-if v1.0.0
│   ├── ndarray v0.16.1 (*)
│   ├── ort-sys v2.0.0-rc.9
│   │   [build-dependencies]
│   │   ├── flate2 v1.0.35 (*)
│   │   ├── pkg-config v0.3.31
│   │   ├── sha2 v0.10.8
│   │   │   ├── cfg-if v1.0.0
│   │   │   ├── cpufeatures v0.2.16
│   │   │   └── digest v0.10.7
│   │   │       ├── block-buffer v0.10.4
│   │   │       │   └── generic-array v0.14.7
│   │   │       │       └── typenum v1.17.0
│   │   │       │       [build-dependencies]
│   │   │       │       └── version_check v0.9.5
│   │   │       └── crypto-common v0.1.6
│   │   │           ├── generic-array v0.14.7 (*)
│   │   │           └── typenum v1.17.0
│   │   ├── tar v0.4.43
│   │   │   ├── filetime v0.2.25
│   │   │   │   ├── cfg-if v1.0.0
│   │   │   │   └── libc v0.2.169
│   │   │   ├── libc v0.2.169
│   │   │   └── xattr v1.4.0
│   │   │       ├── linux-raw-sys v0.4.15
│   │   │       └── rustix v0.38.43
│   │   │           ├── bitflags v2.7.0
│   │   │           └── linux-raw-sys v0.4.15
│   │   └── ureq v2.12.1
│   │       ├── base64 v0.22.1
│   │       ├── log v0.4.25
│   │       ├── once_cell v1.20.2
│   │       ├── rustls v0.23.21 (*)
│   │       ├── rustls-pki-types v1.10.1
│   │       ├── socks v0.3.4
│   │       │   ├── byteorder v1.5.0
│   │       │   └── libc v0.2.169
│   │       ├── url v2.5.4 (*)
│   │       └── webpki-roots v0.26.7 (*)
│   └── tracing v0.1.41 (*)
├── pdf-extract v0.7.7 (https://github.com/jrmuizel/pdf-extract.git?rev=refs%2Fpull%2F91%2Fhead#0b946a5b)
│   ├── adobe-cmap-parser v0.4.1
│   │   └── pom v1.1.0
│   ├── anyhow v1.0.95
│   ├── encoding v0.2.33
│   │   ├── encoding-index-japanese v1.20141219.5
│   │   │   └── encoding_index_tests v0.1.4
│   │   ├── encoding-index-korean v1.20141219.5
│   │   │   └── encoding_index_tests v0.1.4
│   │   ├── encoding-index-simpchinese v1.20141219.5
│   │   │   └── encoding_index_tests v0.1.4
│   │   ├── encoding-index-singlebyte v1.20141219.5
│   │   │   └── encoding_index_tests v0.1.4
│   │   └── encoding-index-tradchinese v1.20141219.5
│   │       └── encoding_index_tests v0.1.4
│   ├── euclid v0.20.14
│   │   └── num-traits v0.2.19 (*)
│   ├── lopdf v0.32.0
│   │   ├── encoding_rs v0.8.35
│   │   │   └── cfg-if v1.0.0
│   │   ├── flate2 v1.0.35 (*)
│   │   ├── itoa v1.0.14
│   │   ├── linked-hash-map v0.5.6
│   │   ├── log v0.4.25
│   │   ├── md5 v0.7.0
│   │   ├── nom v7.1.3 (*)
│   │   ├── time v0.3.37
│   │   │   ├── deranged v0.3.11
│   │   │   │   └── powerfmt v0.2.0
│   │   │   ├── itoa v1.0.14
│   │   │   ├── num-conv v0.1.0
│   │   │   ├── powerfmt v0.2.0
│   │   │   └── time-core v0.1.2
│   │   └── weezl v0.1.8
│   ├── postscript v0.18.4
│   │   └── typeface v0.4.3
│   ├── thiserror v1.0.69 (*)
│   ├── tracing v0.1.41 (*)
│   ├── tracing-subscriber v0.3.19
│   │   ├── matchers v0.1.0
│   │   │   └── regex-automata v0.1.10
│   │   │       └── regex-syntax v0.6.29
│   │   ├── nu-ansi-term v0.46.0
│   │   │   └── overload v0.1.1
│   │   ├── once_cell v1.20.2
│   │   ├── regex v1.11.1
│   │   │   ├── aho-corasick v1.1.3 (*)
│   │   │   ├── memchr v2.7.4
│   │   │   ├── regex-automata v0.4.9 (*)
│   │   │   └── regex-syntax v0.8.5
│   │   ├── sharded-slab v0.1.7
│   │   │   └── lazy_static v1.5.0
│   │   ├── smallvec v1.13.2
│   │   ├── thread_local v1.1.8
│   │   │   ├── cfg-if v1.0.0
│   │   │   └── once_cell v1.20.2
│   │   ├── tracing v0.1.41 (*)
│   │   ├── tracing-core v0.1.33 (*)
│   │   └── tracing-log v0.2.0
│   │       ├── log v0.4.25
│   │       ├── once_cell v1.20.2
│   │       └── tracing-core v0.1.33 (*)
│   ├── type1-encoding-parser v0.1.0
│   │   └── pom v1.1.0
│   └── unicode-normalization v0.1.24
│       └── tinyvec v1.8.1
│           └── tinyvec_macros v0.1.1
├── pdf2image v0.1.2
│   ├── derive_builder v0.20.2
│   │   └── derive_builder_macro v0.20.2 (proc-macro)
│   │       ├── derive_builder_core v0.20.2
│   │       │   ├── darling v0.20.10
│   │       │   │   ├── darling_core v0.20.10
│   │       │   │   │   ├── fnv v1.0.7
│   │       │   │   │   ├── ident_case v1.0.1
│   │       │   │   │   ├── proc-macro2 v1.0.93 (*)
│   │       │   │   │   ├── quote v1.0.38 (*)
│   │       │   │   │   ├── strsim v0.11.1
│   │       │   │   │   └── syn v2.0.96 (*)
│   │       │   │   └── darling_macro v0.20.10 (proc-macro)
│   │       │   │       ├── darling_core v0.20.10 (*)
│   │       │   │       ├── quote v1.0.38 (*)
│   │       │   │       └── syn v2.0.96 (*)
│   │       │   ├── proc-macro2 v1.0.93 (*)
│   │       │   ├── quote v1.0.38 (*)
│   │       │   └── syn v2.0.96 (*)
│   │       └── syn v2.0.96 (*)
│   ├── image v0.25.5 (*)
│   ├── rayon v1.10.0 (*)
│   └── thiserror v1.0.69 (*)
├── rand v0.8.5 (*)
├── rayon v1.10.0 (*)
├── regex v1.11.1 (*)
├── reqwest v0.12.12
│   ├── base64 v0.22.1
│   ├── bytes v1.9.0
│   ├── encoding_rs v0.8.35 (*)
│   ├── futures-channel v0.3.31 (*)
│   ├── futures-core v0.3.31
│   ├── futures-util v0.3.31 (*)
│   ├── h2 v0.4.7
│   │   ├── atomic-waker v1.1.2
│   │   ├── bytes v1.9.0
│   │   ├── fnv v1.0.7
│   │   ├── futures-core v0.3.31
│   │   ├── futures-sink v0.3.31
│   │   ├── http v1.2.0
│   │   │   ├── bytes v1.9.0
│   │   │   ├── fnv v1.0.7
│   │   │   └── itoa v1.0.14
│   │   ├── indexmap v2.7.0 (*)
│   │   ├── slab v0.4.9 (*)
│   │   ├── tokio v1.43.0
│   │   │   ├── bytes v1.9.0
│   │   │   ├── libc v0.2.169
│   │   │   ├── mio v1.0.3
│   │   │   │   └── libc v0.2.169
│   │   │   ├── pin-project-lite v0.2.16
│   │   │   ├── socket2 v0.5.8
│   │   │   │   └── libc v0.2.169
│   │   │   └── tokio-macros v2.5.0 (proc-macro)
│   │   │       ├── proc-macro2 v1.0.93 (*)
│   │   │       ├── quote v1.0.38 (*)
│   │   │       └── syn v2.0.96 (*)
│   │   ├── tokio-util v0.7.13
│   │   │   ├── bytes v1.9.0
│   │   │   ├── futures-core v0.3.31
│   │   │   ├── futures-sink v0.3.31
│   │   │   ├── pin-project-lite v0.2.16
│   │   │   └── tokio v1.43.0 (*)
│   │   └── tracing v0.1.41 (*)
│   ├── http v1.2.0 (*)
│   ├── http-body v1.0.1
│   │   ├── bytes v1.9.0
│   │   └── http v1.2.0 (*)
│   ├── http-body-util v0.1.2
│   │   ├── bytes v1.9.0
│   │   ├── futures-util v0.3.31 (*)
│   │   ├── http v1.2.0 (*)
│   │   ├── http-body v1.0.1 (*)
│   │   └── pin-project-lite v0.2.16
│   ├── hyper v1.5.2
│   │   ├── bytes v1.9.0
│   │   ├── futures-channel v0.3.31 (*)
│   │   ├── futures-util v0.3.31 (*)
│   │   ├── h2 v0.4.7 (*)
│   │   ├── http v1.2.0 (*)
│   │   ├── http-body v1.0.1 (*)
│   │   ├── httparse v1.9.5
│   │   ├── itoa v1.0.14
│   │   ├── pin-project-lite v0.2.16
│   │   ├── smallvec v1.13.2
│   │   ├── tokio v1.43.0 (*)
│   │   └── want v0.3.1
│   │       └── try-lock v0.2.5
│   ├── hyper-tls v0.6.0
│   │   ├── bytes v1.9.0
│   │   ├── http-body-util v0.1.2 (*)
│   │   ├── hyper v1.5.2 (*)
│   │   ├── hyper-util v0.1.10
│   │   │   ├── bytes v1.9.0
│   │   │   ├── futures-channel v0.3.31 (*)
│   │   │   ├── futures-util v0.3.31 (*)
│   │   │   ├── http v1.2.0 (*)
│   │   │   ├── http-body v1.0.1 (*)
│   │   │   ├── hyper v1.5.2 (*)
│   │   │   ├── pin-project-lite v0.2.16
│   │   │   ├── socket2 v0.5.8 (*)
│   │   │   ├── tokio v1.43.0 (*)
│   │   │   ├── tower-service v0.3.3
│   │   │   └── tracing v0.1.41 (*)
│   │   ├── native-tls v0.2.12 (*)
│   │   ├── tokio v1.43.0 (*)
│   │   ├── tokio-native-tls v0.3.1
│   │   │   ├── native-tls v0.2.12 (*)
│   │   │   └── tokio v1.43.0 (*)
│   │   └── tower-service v0.3.3
│   ├── hyper-util v0.1.10 (*)
│   ├── ipnet v2.10.1
│   ├── log v0.4.25
│   ├── mime v0.3.17
│   ├── native-tls v0.2.12 (*)
│   ├── once_cell v1.20.2
│   ├── percent-encoding v2.3.1
│   ├── pin-project-lite v0.2.16
│   ├── rustls-pemfile v2.2.0
│   │   └── rustls-pki-types v1.10.1
│   ├── serde v1.0.217 (*)
│   ├── serde_json v1.0.135 (*)
│   ├── serde_urlencoded v0.7.1
│   │   ├── form_urlencoded v1.2.1 (*)
│   │   ├── itoa v1.0.14
│   │   ├── ryu v1.0.18
│   │   └── serde v1.0.217 (*)
│   ├── sync_wrapper v1.0.2
│   │   └── futures-core v0.3.31
│   ├── tokio v1.43.0 (*)
│   ├── tokio-native-tls v0.3.1 (*)
│   ├── tower v0.5.2
│   │   ├── futures-core v0.3.31
│   │   ├── futures-util v0.3.31 (*)
│   │   ├── pin-project-lite v0.2.16
│   │   ├── sync_wrapper v1.0.2 (*)
│   │   ├── tokio v1.43.0 (*)
│   │   ├── tower-layer v0.3.3
│   │   └── tower-service v0.3.3
│   ├── tower-service v0.3.3
│   └── url v2.5.4 (*)
├── rusty-tesseract v1.1.10
│   ├── image v0.25.5 (*)
│   ├── subprocess v0.2.9
│   │   └── libc v0.2.169
│   ├── substring v1.4.5
│   │   [build-dependencies]
│   │   └── autocfg v1.4.0
│   ├── tempfile v3.15.0
│   │   ├── cfg-if v1.0.0
│   │   ├── fastrand v2.3.0
│   │   ├── getrandom v0.2.15 (*)
│   │   ├── once_cell v1.20.2
│   │   └── rustix v0.38.43
│   │       ├── bitflags v2.7.0
│   │       └── linux-raw-sys v0.4.15
│   └── thiserror v1.0.69 (*)
├── scraper v0.20.0
│   ├── ahash v0.8.11
│   │   ├── cfg-if v1.0.0
│   │   ├── getrandom v0.2.15 (*)
│   │   ├── once_cell v1.20.2
│   │   └── zerocopy v0.7.35 (*)
│   │   [build-dependencies]
│   │   └── version_check v0.9.5
│   ├── cssparser v0.31.2
│   │   ├── cssparser-macros v0.6.1 (proc-macro)
│   │   │   ├── quote v1.0.38 (*)
│   │   │   └── syn v2.0.96 (*)
│   │   ├── dtoa-short v0.3.5
│   │   │   └── dtoa v1.0.9
│   │   ├── itoa v1.0.14
│   │   ├── phf v0.11.3
│   │   │   ├── phf_macros v0.11.3 (proc-macro)
│   │   │   │   ├── phf_generator v0.11.3
│   │   │   │   │   ├── phf_shared v0.11.3
│   │   │   │   │   │   └── siphasher v1.0.1
│   │   │   │   │   └── rand v0.8.5
│   │   │   │   │       ├── libc v0.2.169
│   │   │   │   │       ├── rand_chacha v0.3.1 (*)
│   │   │   │   │       └── rand_core v0.6.4 (*)
│   │   │   │   ├── phf_shared v0.11.3 (*)
│   │   │   │   ├── proc-macro2 v1.0.93 (*)
│   │   │   │   ├── quote v1.0.38 (*)
│   │   │   │   └── syn v2.0.96 (*)
│   │   │   └── phf_shared v0.11.3
│   │   │       └── siphasher v1.0.1
│   │   └── smallvec v1.13.2
│   ├── ego-tree v0.6.3
│   ├── getopts v0.2.21 (*)
│   ├── html5ever v0.27.0
│   │   ├── log v0.4.25
│   │   ├── mac v0.1.1
│   │   └── markup5ever v0.12.1
│   │       ├── log v0.4.25
│   │       ├── phf v0.11.3 (*)
│   │       ├── string_cache v0.8.7
│   │       │   ├── new_debug_unreachable v1.0.6
│   │       │   ├── once_cell v1.20.2
│   │       │   ├── parking_lot v0.12.3
│   │       │   │   ├── lock_api v0.4.12
│   │       │   │   │   └── scopeguard v1.2.0
│   │       │   │   │   [build-dependencies]
│   │       │   │   │   └── autocfg v1.4.0
│   │       │   │   └── parking_lot_core v0.9.10
│   │       │   │       ├── cfg-if v1.0.0
│   │       │   │       ├── libc v0.2.169
│   │       │   │       └── smallvec v1.13.2
│   │       │   ├── phf_shared v0.10.0
│   │       │   │   └── siphasher v0.3.11
│   │       │   ├── precomputed-hash v0.1.1
│   │       │   └── serde v1.0.217 (*)
│   │       └── tendril v0.4.3
│   │           ├── futf v0.1.5
│   │           │   ├── mac v0.1.1
│   │           │   └── new_debug_unreachable v1.0.6
│   │           ├── mac v0.1.1
│   │           └── utf-8 v0.7.6
│   │       [build-dependencies]
│   │       ├── phf_codegen v0.11.3
│   │       │   ├── phf_generator v0.11.3 (*)
│   │       │   └── phf_shared v0.11.3 (*)
│   │       └── string_cache_codegen v0.5.2
│   │           ├── phf_generator v0.10.0
│   │           │   ├── phf_shared v0.10.0 (*)
│   │           │   └── rand v0.8.5 (*)
│   │           ├── phf_shared v0.10.0 (*)
│   │           ├── proc-macro2 v1.0.93 (*)
│   │           └── quote v1.0.38 (*)
│   │   [build-dependencies]
│   │   ├── proc-macro2 v1.0.93 (*)
│   │   ├── quote v1.0.38 (*)
│   │   └── syn v2.0.96 (*)
│   ├── once_cell v1.20.2
│   ├── selectors v0.25.0
│   │   ├── bitflags v2.7.0
│   │   ├── cssparser v0.31.2 (*)
│   │   ├── derive_more v0.99.18 (proc-macro) (*)
│   │   ├── fxhash v0.2.1
│   │   │   └── byteorder v1.5.0
│   │   ├── log v0.4.25
│   │   ├── new_debug_unreachable v1.0.6
│   │   ├── phf v0.10.1
│   │   │   └── phf_shared v0.10.0 (*)
│   │   ├── precomputed-hash v0.1.1
│   │   ├── servo_arc v0.3.0
│   │   │   └── stable_deref_trait v1.2.0
│   │   └── smallvec v1.13.2
│   │   [build-dependencies]
│   │   └── phf_codegen v0.10.0
│   │       ├── phf_generator v0.10.0 (*)
│   │       └── phf_shared v0.10.0 (*)
│   └── tendril v0.4.3 (*)
├── serde v1.0.217 (*)
├── serde_json v1.0.135 (*)
├── statistical v1.0.0
│   ├── num v0.2.1
│   │   ├── num-bigint v0.2.6
│   │   │   ├── num-integer v0.1.46 (*)
│   │   │   └── num-traits v0.2.19 (*)
│   │   │   [build-dependencies]
│   │   │   └── autocfg v1.4.0
│   │   ├── num-complex v0.2.4
│   │   │   └── num-traits v0.2.19 (*)
│   │   │   [build-dependencies]
│   │   │   └── autocfg v1.4.0
│   │   ├── num-integer v0.1.46 (*)
│   │   ├── num-iter v0.1.45 (*)
│   │   ├── num-rational v0.2.4
│   │   │   ├── num-bigint v0.2.6 (*)
│   │   │   ├── num-integer v0.1.46 (*)
│   │   │   └── num-traits v0.2.19 (*)
│   │   │   [build-dependencies]
│   │   │   └── autocfg v1.4.0
│   │   └── num-traits v0.2.19 (*)
│   └── rand v0.6.5
│       ├── libc v0.2.169
│       ├── rand_chacha v0.1.1
│       │   └── rand_core v0.3.1
│       │       └── rand_core v0.4.2
│       │   [build-dependencies]
│       │   └── autocfg v0.1.8
│       │       └── autocfg v1.4.0
│       ├── rand_core v0.4.2
│       ├── rand_hc v0.1.0
│       │   └── rand_core v0.3.1 (*)
│       ├── rand_isaac v0.1.1
│       │   └── rand_core v0.3.1 (*)
│       ├── rand_jitter v0.1.4
│       │   └── rand_core v0.4.2
│       ├── rand_os v0.1.3
│       │   ├── libc v0.2.169
│       │   └── rand_core v0.4.2
│       ├── rand_pcg v0.1.2
│       │   └── rand_core v0.4.2
│       │   [build-dependencies]
│       │   └── autocfg v0.1.8 (*)
│       └── rand_xorshift v0.1.1
│           └── rand_core v0.3.1 (*)
│       [build-dependencies]
│       └── autocfg v0.1.8 (*)
├── strum v0.26.3
│   └── strum_macros v0.26.4 (proc-macro)
│       ├── heck v0.5.0
│       ├── proc-macro2 v1.0.93 (*)
│       ├── quote v1.0.38 (*)
│       ├── rustversion v1.0.19 (proc-macro)
│       └── syn v2.0.96 (*)
├── strum_macros v0.26.4 (proc-macro) (*)
├── text-splitter v0.18.1
│   ├── ahash v0.8.11 (*)
│   ├── auto_enums v0.8.6 (proc-macro)
│   │   ├── derive_utils v0.14.2
│   │   │   ├── proc-macro2 v1.0.93 (*)
│   │   │   ├── quote v1.0.38 (*)
│   │   │   └── syn v2.0.96 (*)
│   │   ├── proc-macro2 v1.0.93 (*)
│   │   ├── quote v1.0.38 (*)
│   │   └── syn v2.0.96 (*)
│   ├── either v1.13.0
│   ├── itertools v0.13.0 (*)
│   ├── once_cell v1.20.2
│   ├── regex v1.11.1 (*)
│   ├── strum v0.26.3 (*)
│   ├── thiserror v1.0.69 (*)
│   ├── tokenizers v0.20.4
│   │   ├── aho-corasick v1.1.3 (*)
│   │   ├── derive_builder v0.20.2 (*)
│   │   ├── esaxx-rs v0.1.10
│   │   │   [build-dependencies]
│   │   │   └── cc v1.2.9 (*)
│   │   ├── getrandom v0.2.15 (*)
│   │   ├── hf-hub v0.3.2 (*)
│   │   ├── indicatif v0.17.9 (*)
│   │   ├── itertools v0.12.1 (*)
│   │   ├── lazy_static v1.5.0
│   │   ├── log v0.4.25
│   │   ├── macro_rules_attribute v0.2.0
│   │   │   ├── macro_rules_attribute-proc_macro v0.2.0 (proc-macro)
│   │   │   └── paste v1.0.15 (proc-macro)
│   │   ├── monostate v0.1.13
│   │   │   ├── monostate-impl v0.1.13 (proc-macro)
│   │   │   │   ├── proc-macro2 v1.0.93 (*)
│   │   │   │   ├── quote v1.0.38 (*)
│   │   │   │   └── syn v2.0.96 (*)
│   │   │   └── serde v1.0.217 (*)
│   │   ├── onig v6.4.0
│   │   │   ├── bitflags v1.3.2
│   │   │   ├── once_cell v1.20.2
│   │   │   └── onig_sys v69.8.1
│   │   │       [build-dependencies]
│   │   │       ├── cc v1.2.9 (*)
│   │   │       └── pkg-config v0.3.31
│   │   ├── paste v1.0.15 (proc-macro)
│   │   ├── rand v0.8.5 (*)
│   │   ├── rayon v1.10.0 (*)
│   │   ├── rayon-cond v0.3.0
│   │   │   ├── either v1.13.0
│   │   │   ├── itertools v0.11.0
│   │   │   │   └── either v1.13.0
│   │   │   └── rayon v1.10.0 (*)
│   │   ├── regex v1.11.1 (*)
│   │   ├── regex-syntax v0.8.5
│   │   ├── serde v1.0.217 (*)
│   │   ├── serde_json v1.0.135 (*)
│   │   ├── spm_precompiled v0.1.4
│   │   │   ├── base64 v0.13.1
│   │   │   ├── nom v7.1.3 (*)
│   │   │   ├── serde v1.0.217 (*)
│   │   │   └── unicode-segmentation v1.12.0
│   │   ├── thiserror v1.0.69 (*)
│   │   ├── unicode-normalization-alignments v0.1.12
│   │   │   └── smallvec v1.13.2
│   │   ├── unicode-segmentation v1.12.0
│   │   └── unicode_categories v0.1.1
│   └── unicode-segmentation v1.12.0
├── tokenizers v0.20.4 (*)
├── tokio v1.43.0 (*)
├── tracing v0.1.41 (*)
├── url v2.5.4 (*)
└── walkdir v2.5.0
    └── same-file v1.0.6
[dev-dependencies]
├── clap v4.5.26 (*)
├── lazy_static v1.5.0
└── tempdir v0.3.7
    ├── rand v0.4.6
    │   └── libc v0.2.169
    └── remove_dir_all v0.5.3

embed_anything_python v0.5.2 (/home/akshay/projects/EmbedAnything/python)
├── embed_anything v0.5.1 (/home/akshay/projects/EmbedAnything/rust) (*)
├── pyo3 v0.23.4
│   ├── cfg-if v1.0.0
│   ├── indoc v2.0.5 (proc-macro)
│   ├── libc v0.2.169
│   ├── memoffset v0.9.1
│   │   [build-dependencies]
│   │   └── autocfg v1.4.0
│   ├── once_cell v1.20.2
│   ├── pyo3-ffi v0.23.4
│   │   └── libc v0.2.169
│   │   [build-dependencies]
│   │   └── pyo3-build-config v0.23.4
│   │       ├── once_cell v1.20.2
│   │       └── target-lexicon v0.12.16
│   │       [build-dependencies]
│   │       └── target-lexicon v0.12.16
│   ├── pyo3-macros v0.23.4 (proc-macro)
│   │   ├── proc-macro2 v1.0.93 (*)
│   │   ├── pyo3-macros-backend v0.23.4
│   │   │   ├── heck v0.5.0
│   │   │   ├── proc-macro2 v1.0.93 (*)
│   │   │   ├── pyo3-build-config v0.23.4 (*)
│   │   │   ├── quote v1.0.38 (*)
│   │   │   └── syn v2.0.96 (*)
│   │   │   [build-dependencies]
│   │   │   └── pyo3-build-config v0.23.4 (*)
│   │   ├── quote v1.0.38 (*)
│   │   └── syn v2.0.96 (*)
│   └── unindent v0.2.3
│   [build-dependencies]
│   └── pyo3-build-config v0.23.4 (*)
├── strum v0.26.3 (*)
├── strum_macros v0.26.4 (proc-macro) (*)
└── tokio v1.43.0 (*)
