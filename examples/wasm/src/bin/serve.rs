//! Serves `src/` over HTTP so `index.html` and the WASM ES module load correctly (not `file://`).
use std::net::SocketAddr;
use std::path::PathBuf;

use axum::Router;
use tokio::net::TcpListener;
use tower_http::services::ServeDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let web_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    if !web_root.join("index.html").is_file() {
        eprintln!(
            "expected {} (run wasm-pack first for src/pkg).",
            web_root.join("index.html").display()
        );
    }

    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr).await?;

    let service = ServeDir::new(&web_root).append_index_html_on_directories(true);
    let app = Router::new().fallback_service(service);

    let url = format!("http://{}", addr);
    eprintln!("Serving {} -> {}", web_root.display(), url);
    eprintln!("Open {url}/ in your browser.");

    axum::serve(listener, app).await?;
    Ok(())
}
