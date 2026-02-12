#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::net::SocketAddr;

use anyhow::Result;
use clap::Parser;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, info_span, Instrument};
use tracing_subscriber::EnvFilter;

use mx8_proto::v0::coordinator_server::{Coordinator, CoordinatorServer};
use mx8_proto::v0::*;

#[derive(Debug, Parser)]
#[command(name = "mx8-coordinator")]
struct Args {
    /// Address to bind the coordinator gRPC server.
    #[arg(long, env = "MX8_COORD_BIND_ADDR", default_value = "0.0.0.0:50051")]
    addr: SocketAddr,
}

#[derive(Debug, Default)]
struct CoordinatorSvc;

#[tonic::async_trait]
impl Coordinator for CoordinatorSvc {
    async fn register_node(
        &self,
        _request: Request<RegisterNodeRequest>,
    ) -> Result<Response<RegisterNodeResponse>, Status> {
        Err(Status::unimplemented("register_node not implemented"))
    }

    async fn heartbeat(
        &self,
        _request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        Err(Status::unimplemented("heartbeat not implemented"))
    }

    async fn request_lease(
        &self,
        _request: Request<RequestLeaseRequest>,
    ) -> Result<Response<RequestLeaseResponse>, Status> {
        Err(Status::unimplemented("request_lease not implemented"))
    }

    async fn report_progress(
        &self,
        _request: Request<ReportProgressRequest>,
    ) -> Result<Response<ReportProgressResponse>, Status> {
        Err(Status::unimplemented("report_progress not implemented"))
    }

    async fn get_manifest(
        &self,
        _request: Request<GetManifestRequest>,
    ) -> Result<Response<GetManifestResponse>, Status> {
        Err(Status::unimplemented("get_manifest not implemented"))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let span = info_span!("mx8-coordinator", addr = %args.addr);
    async move {
        info!("starting coordinator (v0 skeleton)");
        Server::builder()
            .add_service(CoordinatorServer::new(CoordinatorSvc))
            .serve(args.addr)
            .await?;
        Ok(())
    }
    .instrument(span)
    .await
}
