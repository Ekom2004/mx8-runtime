from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mx8


def _consume_ids(loader: mx8.DistributedDataLoader, max_batches: int | None) -> list[int]:
    out: list[int] = []
    batches = 0
    while True:
        if max_batches is not None and batches >= max_batches:
            break
        try:
            batch = next(loader)
        except StopIteration:
            break
        out.extend(int(v) for v in batch.sample_ids)
        batches += 1
    return out


def _consume_until(
    loader: mx8.DistributedDataLoader,
    target_samples: int,
    max_batches: int,
) -> list[int]:
    out: list[int] = []
    batches = 0
    while len(out) < target_samples:
        if batches >= max_batches:
            raise SystemExit(
                f"resume did not reach target samples in {max_batches} batches: got={len(out)} target={target_samples}"
            )
        try:
            batch = next(loader)
        except StopIteration:
            break
        out.extend(int(v) for v in batch.sample_ids)
        batches += 1
    return out


def _new_loader(
    *,
    coord_url: str,
    job_id: str,
    node_id: str,
    batch_size_samples: int,
    want: int,
    progress_interval_ms: int,
    resume_from: bytes | None,
) -> mx8.DistributedDataLoader:
    return mx8.DistributedDataLoader(
        coord_url=coord_url,
        job_id=job_id,
        node_id=node_id,
        batch_size_samples=batch_size_samples,
        max_inflight_bytes=int(os.environ.get("MX8_MAX_INFLIGHT_BYTES", str(8 * 1024 * 1024))),
        max_queue_batches=int(os.environ.get("MX8_MAX_QUEUE_BATCHES", "8")),
        prefetch_batches=int(os.environ.get("MX8_PREFETCH_BATCHES", "1")),
        want=want,
        progress_interval_ms=progress_interval_ms,
        resume_from=resume_from,
    )


def _capture(args: argparse.Namespace) -> None:
    loader = _new_loader(
        coord_url=args.coord_url,
        job_id=args.job_id,
        node_id=args.node_id,
        batch_size_samples=args.batch_size_samples,
        want=args.want,
        progress_interval_ms=args.progress_interval_ms,
        resume_from=None,
    )
    try:
        seen = _consume_ids(loader, max_batches=args.phase1_batches)
        if not seen:
            raise SystemExit("capture produced zero samples")
        if len(set(seen)) != len(seen):
            raise SystemExit("capture observed duplicate sample ids")
        token = bytes(loader.checkpoint())
        if not token:
            raise SystemExit("capture checkpoint token is empty")
    finally:
        loader.close()

    Path(args.token_path).write_bytes(token)
    Path(args.seen_path).write_text(json.dumps({"seen_ids": seen}), encoding="utf-8")
    print("capture_ids:", len(seen))
    print("capture_token_bytes:", len(token))


def _resume(args: argparse.Namespace) -> None:
    token = Path(args.token_path).read_bytes()
    seen_phase1 = json.loads(Path(args.seen_path).read_text(encoding="utf-8"))["seen_ids"]
    seen_phase1 = [int(v) for v in seen_phase1]
    phase1_set = set(seen_phase1)
    if len(phase1_set) != len(seen_phase1):
        raise SystemExit("phase1 snapshot contains duplicates")

    loader = _new_loader(
        coord_url=args.coord_url,
        job_id=args.job_id,
        node_id=args.node_id,
        batch_size_samples=args.batch_size_samples,
        want=args.want,
        progress_interval_ms=args.progress_interval_ms,
        resume_from=token,
    )
    try:
        expected_remaining = args.total_samples - len(phase1_set)
        if expected_remaining <= 0:
            raise SystemExit("phase1 already covers total samples; invalid gate config")
        seen_phase2 = _consume_until(loader, expected_remaining, args.max_batches)
    finally:
        loader.close()

    if len(seen_phase2) != expected_remaining:
        raise SystemExit(
            f"resume sample count mismatch: expected={expected_remaining} got={len(seen_phase2)}"
        )

    phase2_set = set(seen_phase2)
    if len(phase2_set) != len(seen_phase2):
        raise SystemExit("phase2 observed duplicate sample ids")
    overlap = phase1_set.intersection(phase2_set)
    if overlap:
        first = min(overlap)
        raise SystemExit(f"resume overlap detected at sample_id={first}")

    union = phase1_set.union(phase2_set)
    if len(union) != args.total_samples:
        raise SystemExit(
            f"resume coverage mismatch: expected={args.total_samples} got={len(union)}"
        )
    if min(union) != 0 or max(union) != args.total_samples - 1:
        raise SystemExit("resume sample-id bounds mismatch")

    print("phase1_ids:", len(seen_phase1))
    print("phase2_ids:", len(seen_phase2))
    print("union_ids:", len(union))
    print("overlap_ids:", len(overlap))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MX8 distributed checkpoint/resume gate")
    p.add_argument("--mode", choices=("capture", "resume"), required=True)
    p.add_argument("--coord-url", default=os.environ.get("MX8_COORD_URL", "http://127.0.0.1:50051"))
    p.add_argument("--job-id", default=os.environ.get("MX8_JOB_ID", "m5-distributed-resume-gate"))
    p.add_argument("--node-id", default=os.environ.get("MX8_NODE_ID", "rank0"))
    p.add_argument("--batch-size-samples", type=int, default=int(os.environ.get("MX8_BATCH_SIZE_SAMPLES", "128")))
    p.add_argument("--phase1-batches", type=int, default=int(os.environ.get("MX8_PHASE1_BATCHES", "8")))
    p.add_argument("--want", type=int, default=int(os.environ.get("MX8_DEV_LEASE_WANT", "1")))
    p.add_argument("--progress-interval-ms", type=int, default=int(os.environ.get("MX8_PROGRESS_INTERVAL_MS", "100")))
    p.add_argument("--total-samples", type=int, default=int(os.environ.get("MX8_TOTAL_SAMPLES", "4096")))
    p.add_argument("--max-batches", type=int, default=int(os.environ.get("MX8_MAX_BATCHES", "10000")))
    p.add_argument("--token-path", required=True)
    p.add_argument("--seen-path", required=True)
    args = p.parse_args()
    if args.total_samples <= 0:
        raise SystemExit("total-samples must be > 0")
    if args.batch_size_samples <= 0:
        raise SystemExit("batch-size-samples must be > 0")
    if args.phase1_batches <= 0:
        raise SystemExit("phase1-batches must be > 0")
    if args.want <= 0:
        raise SystemExit("want must be > 0")
    if args.max_batches <= 0:
        raise SystemExit("max-batches must be > 0")
    return args


def main() -> None:
    args = _parse_args()
    if args.mode == "capture":
        _capture(args)
        return
    _resume(args)


if __name__ == "__main__":
    main()
