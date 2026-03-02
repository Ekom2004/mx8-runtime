from __future__ import annotations

import argparse
import json
import os
import time
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


def _completed_sample_count_from_token(token: bytes) -> int:
    total = 0
    for raw in token.decode("utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("C "):
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        start = int(parts[1])
        end = int(parts[2])
        if end > start:
            total += end - start
    return total


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
        # Checkpoint tokens encode only fully completed ranges (no partial range cursor).
        # In-flight progress RPCs can lag briefly, so wait for token coverage to catch up.
        deadline = time.monotonic() + max(2.0, (args.progress_interval_ms / 1000.0) * 20.0)
        sleep_s = max(0.02, min(0.2, args.progress_interval_ms / 1000.0))
        token = b""
        covered = 0
        while True:
            token = bytes(loader.checkpoint())
            if not token:
                raise SystemExit("capture checkpoint token is empty")
            covered = _completed_sample_count_from_token(token)
            if covered >= len(seen):
                break
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"checkpoint coverage lagged behind consumed samples: covered={covered} seen={len(seen)}"
                )
            time.sleep(sleep_s)
    finally:
        loader.close()

    Path(args.token_path).write_bytes(token)
    Path(args.seen_path).write_text(json.dumps({"seen_ids": seen}), encoding="utf-8")
    print("capture_ids:", len(seen))
    print("capture_token_bytes:", len(token))


def _resume(args: argparse.Namespace) -> None:
    token = Path(args.token_path).read_bytes()
    token_covered = _completed_sample_count_from_token(token)
    if token_covered <= 0:
        raise SystemExit("resume token must cover at least one sample")
    if token_covered > args.total_samples:
        raise SystemExit(
            f"resume token covers more samples than dataset: covered={token_covered} total={args.total_samples}"
        )

    seen_phase1 = json.loads(Path(args.seen_path).read_text(encoding="utf-8"))["seen_ids"]
    seen_phase1 = [int(v) for v in seen_phase1]
    phase1_set = set(seen_phase1)
    if len(phase1_set) != len(seen_phase1):
        raise SystemExit("phase1 snapshot contains duplicates")

    checkpoint_set = set(range(token_covered))
    if not phase1_set.issubset(checkpoint_set):
        raise SystemExit(
            f"phase1 observed ids beyond checkpoint boundary: max_phase1={max(phase1_set)} covered={token_covered}"
        )

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
        expected_remaining = args.total_samples - token_covered
        if expected_remaining < 0:
            raise SystemExit("invalid resume config: token covers past total samples")
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
    overlap = checkpoint_set.intersection(phase2_set)
    if overlap:
        first = min(overlap)
        raise SystemExit(f"resume overlap with checkpoint boundary at sample_id={first}")

    union = checkpoint_set.union(phase2_set)
    if len(union) != args.total_samples:
        raise SystemExit(
            f"resume coverage mismatch: expected={args.total_samples} got={len(union)}"
        )
    if min(union) != 0 or max(union) != args.total_samples - 1:
        raise SystemExit("resume sample-id bounds mismatch")

    print("phase1_ids:", len(seen_phase1))
    print("checkpoint_ids:", token_covered)
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
