# Deployment Guide

This guide explains how to run a distributed MX8 job. A typical deployment has one `mx8-coordinator` process acting as the job control plane and one `mx8d-agent` process per node. The coordinator manages leases and dataset ownership. Each agent enforces local memory caps and feeds work ranges to the runtimes on that machine.

Primary implementations: `crates/mx8-coordinator/src/main.rs`, `crates/mx8d-agent/src/main.rs`.


## What you need to know before deploying

The coordinator is a single process per job. In v1.8 there is no automatic failover — if it dies, the job control plane pauses until it is restarted. Run it on a stable node, not a spot instance. See `docs/ha_contract.md` for the v1.9 HA plan.

Training is non-elastic in v1.8. If a DDP rank dies, the training job terminates. Lease reassignment handles inference and ETL recovery, not training node loss.

The coordinator only serves control-plane traffic and manifest bytes. It never proxies dataset data bytes. Agents and runtimes fetch data directly from storage.


## Topology

The coordinator binds a gRPC port (default `0.0.0.0:50051`) that every agent connects to. Each agent runs on one node and talks to the coordinator for leases, heartbeats, and progress reporting. The manifest store is a shared backend — either an S3 prefix or a filesystem path — that the coordinator reads and writes, and that agents can access directly or receive from the coordinator via proxy.

Place the coordinator on a node with a stable network path to every agent and to the manifest store. The default heartbeat interval is 1 second, so network reliability matters more than raw latency.


## Startup order

Build the binaries first, then start the coordinator, confirm it is listening, then start one agent per node.

```bash
cargo build -p mx8-coordinator --features s3
cargo build -p mx8d-agent --features s3
```

If you are using filesystem-only datasets and a local manifest store, omit `--features s3`.

Start the coordinator:

```bash
MX8_COORD_BIND_ADDR=0.0.0.0:50051 \
MX8_WORLD_SIZE=8 \
MX8_DATASET_LINK='s3://your-bucket/dataset/@refresh' \
MX8_MANIFEST_STORE_ROOT='s3://your-bucket/manifests/' \
target/release/mx8-coordinator
```

Start one agent on each node:

```bash
MX8_COORD_URL='http://coordinator-host:50051' \
MX8_JOB_ID='prod-job-001' \
MX8_NODE_ID='node-0' \
MX8_BATCH_SIZE_SAMPLES=512 \
MX8_MAX_INFLIGHT_BYTES=134217728 \
MX8_MAX_PROCESS_RSS_BYTES=68719476736 \
target/release/mx8d-agent
```

To verify the job is running correctly, open the TUI:

```bash
cargo run -p mx8-tui -- --coord-url http://coordinator-host:50051 --job-id prod-job-001
```

For a deterministic local recovery test: `./scripts/demo2_local.sh`. For distributed S3-compatible validation using MinIO: `./scripts/demo2_minio.sh`.


## Coordinator configuration

Every CLI argument has a corresponding environment variable. All timing values are in milliseconds.

| CLI argument | Environment variable | Default | Notes |
| --- | --- | --- | --- |
| `--addr` | `MX8_COORD_BIND_ADDR` | `0.0.0.0:50051` | gRPC bind address |
| `--world-size` | `MX8_WORLD_SIZE` | `1` | membership barrier; job freezes at this count |
| `--heartbeat-interval-ms` | `MX8_HEARTBEAT_INTERVAL_MS` | `1000` | returned to agents as the heartbeat cadence |
| `--lease-ttl-ms` | `MX8_LEASE_TTL_MS` | `10000` | how long before a silent lease is expired |
| `--dataset-link` | `MX8_DATASET_LINK` | unset | plain path, `@refresh`, or `@sha256:<hash>` |
| `--manifest-store-root` | `MX8_MANIFEST_STORE_ROOT` | `/var/lib/mx8/manifests` | filesystem path or S3 prefix |
| `--snapshot-lock-stale-ms` | `MX8_SNAPSHOT_LOCK_STALE_MS` | `60000` | stale threshold for snapshot locks |
| `--snapshot-wait-timeout-ms` | `MX8_SNAPSHOT_WAIT_TIMEOUT_MS` | `30000` | how long to wait for a concurrent indexer |
| `--shuffle` | `MX8_SHUFFLE` | `false` | enable deterministic block shuffle |
| `--seed` | `MX8_SEED` | `0` | shuffle seed |
| `--epoch` | `MX8_EPOCH` | `0` | epoch input for deterministic ordering |
| `--grpc-max-message-bytes` | `MX8_GRPC_MAX_MESSAGE_BYTES` | `67108864` | gRPC message size cap (64MB) |
| `--metrics-snapshot-interval-ms` | `MX8_METRICS_SNAPSHOT_INTERVAL_MS` | `0` | set to non-zero to enable periodic metrics logging |


## Agent configuration

| CLI argument | Environment variable | Default | Notes |
| --- | --- | --- | --- |
| `--coord-url` | `MX8_COORD_URL` | `http://127.0.0.1:50051` | coordinator address |
| `--job-id` | `MX8_JOB_ID` | `local-job` | must match coordinator job |
| `--node-id` | `MX8_NODE_ID` | `local-node` | stable identity for this node |
| `--batch-size-samples` | `MX8_BATCH_SIZE_SAMPLES` | `512` | samples per batch |
| `--prefetch-batches` | `MX8_PREFETCH_BATCHES` | `1` | pipeline read-ahead depth |
| `--max-queue-batches` | `MX8_MAX_QUEUE_BATCHES` | `64` | delivered queue cap |
| `--max-inflight-bytes` | `MX8_MAX_INFLIGHT_BYTES` | `134217728` | pipeline byte budget (128MB) |
| `--max-process-rss-bytes` | `MX8_MAX_PROCESS_RSS_BYTES` | unset | whole-process RSS fail-fast cap; set by policy |
| `--progress-interval-ms` | `MX8_PROGRESS_INTERVAL_MS` | `500` | how often to report cursor progress |
| `--grpc-max-message-bytes` | `MX8_GRPC_MAX_MESSAGE_BYTES` | `67108864` | must match coordinator setting |
| `--target-batch-bytes` | `MX8_TARGET_BATCH_BYTES` | unset | optional byte target per batch |
| `--max-batch-bytes` | `MX8_MAX_BATCH_BYTES` | unset | optional hard byte cap per batch |

`MX8_AGENT_MANIFEST_STREAM_MAX_LINE_BYTES` (default 8MB) limits the tolerated line length when streaming manifest chunks from the coordinator. Increase only if you have exceptionally wide manifest rows.


## Failure and restart behavior

When an agent dies or stops heartbeating, the coordinator waits for the lease TTL to expire (default 10 seconds), then requeues the unfinished range `[cursor, end)`. The remaining agents pick it up and drain continues. No manual intervention is needed unless the agent does not come back.

To restart a failed agent, bring it back up with the same `MX8_JOB_ID` and `MX8_NODE_ID`. Verify in the TUI that heartbeats resume and the requeued range is picked up.

If the coordinator dies, the job control plane pauses. Restart it, then verify that agents reconnect and heartbeats resume. If the job does not resume cleanly after restart, relaunch from the pinned snapshot using `@sha256:<manifest_hash>`.


## Production baseline

Before running any production job, set these explicitly on every deployment:

`MX8_MAX_PROCESS_RSS_BYTES` should be set as an org-wide policy default, not left to per-job ad-hoc values. Use `MX8_WORLD_SIZE` explicitly rather than relying on the default. Use `MX8_DATASET_LINK` with `@refresh` or a pinned `@sha256:` hash — never rely on an unpinned plain prefix for production. Back `MX8_MANIFEST_STORE_ROOT` by S3 or durable disk, not local ephemeral storage. Set `MX8_GRPC_MAX_MESSAGE_BYTES` to the same value on coordinator and all agents.

Run the readiness suite before any new deployment:

```bash
./scripts/prod_readiness.sh
```
