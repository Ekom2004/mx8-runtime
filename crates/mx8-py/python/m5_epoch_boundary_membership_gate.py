from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
from dataclasses import dataclass


@dataclass
class WorkerResult:
    node_id: str
    ids: list[int]
    error: str | None


def _worker(
    *,
    coord_url: str,
    job_id: str,
    node_id: str,
    batch_size_samples: int,
    want: int,
    progress_interval_ms: int,
    max_inflight_bytes: int,
    max_queue_batches: int,
    prefetch_batches: int,
    steps: int,
    out_queue: mp.Queue,
) -> None:
    import mx8

    loader = mx8.DistributedDataLoader(
        coord_url=coord_url,
        job_id=job_id,
        node_id=node_id,
        batch_size_samples=batch_size_samples,
        max_inflight_bytes=max_inflight_bytes,
        max_queue_batches=max_queue_batches,
        prefetch_batches=prefetch_batches,
        want=want,
        progress_interval_ms=progress_interval_ms,
    )
    seen: list[int] = []
    err: str | None = None
    try:
        for _ in range(steps):
            batch = next(loader)
            seen.extend(int(v) for v in batch.sample_ids)
        if not seen:
            err = f"node {node_id} yielded zero samples"
        if len(seen) != len(set(seen)):
            err = f"duplicate sample IDs observed within node {node_id}"
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    finally:
        loader.close()
    out_queue.put(WorkerResult(node_id=node_id, ids=seen, error=err))


def _validate(results: list[WorkerResult]) -> None:
    all_seen: set[int] = set()
    total_ids = 0
    for result in results:
        if result.error:
            raise SystemExit(f"worker {result.node_id} failed: {result.error}")
        total_ids += len(result.ids)
        ids_set = set(result.ids)
        overlap = all_seen.intersection(ids_set)
        if overlap:
            first = min(overlap)
            raise SystemExit(
                f"cross-node overlap detected at sample_id={first} (node={result.node_id})"
            )
        all_seen.update(ids_set)

    print("workers:", len(results))
    print("total_ids_seen:", total_ids)
    print("union_ids:", len(all_seen))
    print("overlap_ids:", 0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MX8 distributed epoch gate (fixed-step no-overlap)"
    )
    p.add_argument("--coord-url", default=os.environ.get("MX8_COORD_URL", "http://127.0.0.1:50051"))
    p.add_argument("--job-id", default=os.environ.get("MX8_JOB_ID", "m5-epoch-boundary"))
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "2")))
    p.add_argument("--node-prefix", default=os.environ.get("MX8_NODE_PREFIX", "rank"))
    p.add_argument("--steps", type=int, default=int(os.environ.get("MX8_EPOCH_GATE_STEPS", "8")))
    p.add_argument("--batch-size-samples", type=int, default=int(os.environ.get("MX8_BATCH_SIZE_SAMPLES", "64")))
    p.add_argument("--want", type=int, default=int(os.environ.get("MX8_DEV_LEASE_WANT", "1")))
    p.add_argument("--progress-interval-ms", type=int, default=int(os.environ.get("MX8_PROGRESS_INTERVAL_MS", "100")))
    p.add_argument("--max-inflight-bytes", type=int, default=int(os.environ.get("MX8_MAX_INFLIGHT_BYTES", str(8 * 1024 * 1024))))
    p.add_argument("--max-queue-batches", type=int, default=int(os.environ.get("MX8_MAX_QUEUE_BATCHES", "16")))
    p.add_argument("--prefetch-batches", type=int, default=int(os.environ.get("MX8_PREFETCH_BATCHES", "2")))
    args = p.parse_args()
    if args.world_size <= 0:
        raise SystemExit("world-size must be > 0")
    if args.steps <= 0:
        raise SystemExit("steps must be > 0")
    if args.batch_size_samples <= 0:
        raise SystemExit("batch-size-samples must be > 0")
    if args.want <= 0:
        raise SystemExit("want must be > 0")
    return args


def main() -> None:
    args = _parse_args()
    ctx = mp.get_context("spawn")
    out_queue: mp.Queue = ctx.Queue()
    workers: list[mp.Process] = []
    for rank in range(args.world_size):
        node_id = f"{args.node_prefix}{rank}"
        p = ctx.Process(
            target=_worker,
            kwargs={
                "coord_url": args.coord_url,
                "job_id": args.job_id,
                "node_id": node_id,
                "batch_size_samples": args.batch_size_samples,
                "want": args.want,
                "progress_interval_ms": args.progress_interval_ms,
                "max_inflight_bytes": args.max_inflight_bytes,
                "max_queue_batches": args.max_queue_batches,
                "prefetch_batches": args.prefetch_batches,
                "steps": args.steps,
                "out_queue": out_queue,
            },
            daemon=False,
        )
        p.start()
        workers.append(p)

    results: list[WorkerResult] = []
    for _ in workers:
        try:
            result = out_queue.get(timeout=300)
        except queue.Empty as exc:
            raise SystemExit("timed out waiting for worker results") from exc
        results.append(result)

    for p in workers:
        p.join(timeout=30)
        if p.exitcode not in (0, None):
            raise SystemExit(f"worker process exited non-zero: pid={p.pid} code={p.exitcode}")

    _validate(results)


if __name__ == "__main__":
    main()
