use anyhow::Result;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// Starts a simple HTTP dashboard server for cluster health monitoring.
///
/// The server exposes:
/// - GET /metrics -> plain text metrics (same as `metrics::get_metrics()`)
/// - GET / -> simple HTML dashboard linking to /metrics
pub async fn run_dashboard(addr: &str, shutdown: oneshot::Receiver<()>) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(%addr, "Dashboard server started");

    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            _ = &mut shutdown => {
                tracing::info!(%addr, "Dashboard server shutting down");
                break;
            }
            accept = listener.accept() => {
                let (mut socket, peer) = accept?;
                let metrics = crate::metrics::get_metrics();
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let n = match socket.read(&mut buf).await {
                        Ok(n) => n,
                        Err(_) => return,
                    };

                    let request = String::from_utf8_lossy(&buf[..n]);
                    let request_line = request.lines().next().unwrap_or_default();
                    let parts: Vec<&str> = request_line.split_whitespace().collect();
                    let (method, path) = if parts.len() >= 2 {
                        (parts[0], parts[1])
                    } else {
                        ("", "")
                    };

                    let (status, body, content_type) = match (method, path) {
                        ("GET", "/metrics") => ("200 OK", metrics.clone(), "text/plain; charset=utf-8"),
                        ("GET", "/") => (
                            "200 OK",
                            format!(
                                "<html><head><title>Swarm Inference Dashboard</title></head><body><h1>Swarm Inference Dashboard</h1><p><a href=\"/metrics\">View metrics</a></p><pre>{}</pre></body></html>",
                                metrics
                            ),
                            "text/html; charset=utf-8",
                        ),
                        _ => ("404 Not Found", "Not found".to_string(), "text/plain; charset=utf-8"),
                    };

                    let response = format!(
                        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );

                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.shutdown().await;
                    tracing::debug!(%peer, %path, "Served dashboard request");
                });
            }
        }
    }

    Ok(())
}
