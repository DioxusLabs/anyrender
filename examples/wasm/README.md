## Build with `wasm-pack`

From the repo root:

```bash
# one-time
rustup toolchain install 1.92.0
rustup target add wasm32-unknown-unknown --toolchain 1.92.0
cargo +1.92.0 install wasm-pack

# build into examples/wasm/src/pkg so index.html can import it
wasm-pack build examples/wasm --target web --out-dir src/pkg
```

## Run the demo

Serve the `examples/wasm/src` folder (needs a web server; `file://` won't work).

```bash
cargo run -p anyrender_wasm_example --bin serve -- 8080
```

Then open `http://localhost:8080` and click **Paint** (or just edit the HTML).
