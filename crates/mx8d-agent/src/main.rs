#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tonic::transport::Channel;
use tracing::{info, info_span, Instrument};
use tracing_subscriber::EnvFilter;

use mx8_proto::v0::coordinator_client::CoordinatorClient;
use mx8_proto::v0::NodeCaps;
use mx8_proto::v0::RegisterNodeRequest;

#[derive(Debug, Parser)]
#[command(name = "mx8d-agent")]
struct Args {
    /// Coordinator address, e.g. http://127.0.0.1:50051
    #[arg(long, env = "MX8_COORD_URL", default_value = "http://127.0.0.1:50051")]
    coord_url: String,

    #[arg(long, env = "MX8_JOB_ID", default_value = "local-job")]
    job_id: String,

    #[arg(long, env = "MX8_NODE_ID", default_value = "local-node")]
    node_id: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let span = info_span!(
        "mx8d-agent",
        job_id = %args.job_id,
        node_id = %args.node_id,
        coord_url = %args.coord_url
    );

    async move {
        info!("starting agent (v0 skeleton)");
        let channel = Channel::from_shared(args.coord_url)?.connect().await?;
        let mut client = CoordinatorClient::new(channel);

        let _ = client
            .register_node(RegisterNodeRequest {
                job_id: args.job_id.clone(),
                node_id: args.node_id.clone(),
                caps: Some(NodeCaps {
                    max_fetch_concurrency: 32,
                    max_decode_concurrency: 8,
                    max_inflight_bytes: 1 << 30,
                    max_ram_bytes: 4 << 30,
                }),
            })
            .await;

        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            info!("agent heartbeat placeholder");
        }
    }
    .instrument(span)
    .await
}
