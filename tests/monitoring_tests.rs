use std::net::TcpListener;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::oneshot;

/// Find an available TCP port by binding to port 0 and returning the assigned port.
fn get_free_port() -> u16 {
    TcpListener::bind(("127.0.0.1", 0))
        .expect("Failed to bind to ephemeral port")
        .local_addr()
        .unwrap()
        .port()
}

#[tokio::test]
async fn test_dashboard_serves_metrics() {
    // Ensure metrics has some known values
    swarm_inference::metrics::MESSAGES_RECEIVED.store(42, std::sync::atomic::Ordering::Relaxed);

    let port = get_free_port();
    let addr = format!("127.0.0.1:{}", port);

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    // Start the dashboard server
    let server_addr = addr.clone();
    let server_handle = tokio::spawn(async move {
        swarm_inference::dashboard::run_dashboard(&server_addr, shutdown_rx)
            .await
            .expect("Dashboard should start");
    });

    // Give the server time to start
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Fetch metrics over HTTP
    let mut stream = tokio::net::TcpStream::connect(&addr)
        .await
        .expect("Failed to connect to dashboard");

    let request = format!("GET /metrics HTTP/1.1\r\nHost: {}\r\n\r\n", addr);
    stream
        .write_all(request.as_bytes())
        .await
        .expect("Failed to send request");

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .expect("Failed to read response");
    let response = String::from_utf8_lossy(&response);

    assert!(response.contains("messages_received: 42"));

    // Shut down the server
    let _ = shutdown_tx.send(());
    let _ = server_handle.await;
}
