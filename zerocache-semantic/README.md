# zerocache-semantic

CPU text embedder (candle + all-MiniLM-L6-v2) and an in-memory per-scope HNSW
index, for the opt-in semantic completion tier. Not a workspace member — pulled
in only by `zerocache-http --features semantic`.

## Model

`model/all-MiniLM-L6-v2.f16.safetensors` (~45 MB), `tokenizer.json`, `config.json`
are committed and compiled into the binary via `include_bytes!`, so the crate
builds with no network. Source: `sentence-transformers/all-MiniLM-L6-v2`
(Apache-2.0), f32 weights cast to f16.

Regenerate:

```sh
cd scripts && uv run --with safetensors --with numpy python convert-model.py
```

## Build / test

```sh
cargo test --manifest-path zerocache-semantic/Cargo.toml
```
