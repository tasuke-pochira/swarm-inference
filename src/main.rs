// Swarm Inference Protocol
// Developed by Tasuke Pochira, independent developer
// Licensed under Apache 2.0

mod auto_scaling;
mod checkpoint;
mod config;
mod coordinator;
mod dashboard;
mod erasure;
mod gpu;
mod kv_cache;
mod metrics;
mod model;
mod network;
mod node;
mod numa;

extern crate swarm_inference;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use swarm_inference::{AuditResult, get_audit_logger};
use tokio::sync::oneshot;

#[derive(Parser)]
#[command(name = "swarm_inference")]
#[command(about = "Heterogeneous Swarm Inference Protocol")]
struct Cli {
    #[arg(short, long)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Node {
        #[arg(short, long)]
        addr: Option<String>,
        #[arg(short, long)]
        id: usize,
        #[arg(short, long)]
        next: Option<String>,
        #[arg(short, long)]
        coordinator: Option<String>,
    },
    Coordinator {
        #[arg(short, long)]
        first_node: Option<String>,
        #[arg(short, long)]
        listen: Option<String>,
        #[arg(short, long)]
        prompt: String,
    },
    Metrics,
    Dashboard {
        #[arg(short, long)]
        addr: Option<String>,
    },
    Benchmark,
}

fn main() -> Result<()> {
    // Parse CLI early so errors are shown before runtime startup.
    let cli = Cli::parse();

    // Load configuration
    let config = crate::config::Config::load(cli.config.as_deref())
        .map_err(|e| anyhow::anyhow!("Configuration error: {}", e))?;
    config
        .validate()
        .map_err(|e| anyhow::anyhow!("Configuration validation error: {}", e))?;

    // Initialize tracing based on config
    let tracing_level = match config.monitoring.tracing_level.as_str() {
        "error" => tracing::Level::ERROR,
        "warn" => tracing::Level::WARN,
        "info" => tracing::Level::INFO,
        "debug" => tracing::Level::DEBUG,
        "trace" => tracing::Level::TRACE,
        _ => tracing::Level::INFO,
    };

    tracing_subscriber::fmt()
        .with_max_level(tracing_level)
        .init();

    // Initialize audit logging
    let system_id = format!("swarm-{}", std::process::id());
    let node_id = match &cli.command {
        Commands::Node { id, .. } => format!("node-{}", id),
        Commands::Coordinator { .. } => "coordinator".to_string(),
        Commands::Metrics => "metrics".to_string(),
        Commands::Dashboard { .. } => "dashboard".to_string(),
        Commands::Benchmark => "benchmark".to_string(),
    };
    swarm_inference::audit::init_audit_logger(node_id, system_id);

    // Build a Tokio runtime that pins worker threads when requested via env.
    let cores = numa::parse_cores_from_env();
    let mut runtime_builder = tokio::runtime::Builder::new_multi_thread();
    runtime_builder.enable_all();

    if cores.is_some() {
        runtime_builder.on_thread_start(|| {
            // Note: this will be invoked on each worker thread.
            numa::pin_thread_from_env();
        });
    }

    let runtime = runtime_builder.build()?;

    runtime.block_on(async move {
        match cli.command {
            Commands::Node {
                addr,
                id,
                next,
                coordinator,
            } => {
                let addr = addr.unwrap_or_else(|| config.network.listen_addr.clone());
                let coordinator =
                    coordinator.unwrap_or_else(|| config.network.coordinator_addr.clone());
                let node = node::Node::new(id, next, coordinator);
                node.run(&addr).await?;
            }
            Commands::Coordinator {
                first_node,
                listen,
                prompt,
            } => {
                let first_node =
                    first_node.unwrap_or_else(|| config.network.coordinator_addr.clone());
                let listen = listen.unwrap_or_else(|| config.network.listen_addr.clone());
                let coord = coordinator::Coordinator::new(first_node, listen, &config).await?;

                // Start auto-scaling service in background if enabled
                coord.start_scaling_service().await?;

                let result = coord.run_inference(&prompt).await?;
                println!("Result: {:?}", result);
            }
            Commands::Metrics => {
                println!("{}", metrics::get_metrics());
            }
            Commands::Dashboard { addr } => {
                let addr = addr.unwrap_or_else(|| config.monitoring.dashboard_addr.clone());
                let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
                tokio::spawn(async move {
                    let _ = tokio::signal::ctrl_c().await;
                    let _ = shutdown_tx.send(());
                });
                dashboard::run_dashboard(&addr, shutdown_rx).await?;
            }
            Commands::Benchmark => {
                swarm_inference::benchmark::run_all_benchmarks().await;
            }
        }

        Ok(())
    })
}
