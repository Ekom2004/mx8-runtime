from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import queue as py_queue
import traceback

import mx8


def _source_tag(batch) -> str:
    offsets = batch.offsets
    payload = batch.payload
    if len(offsets) < 2:
        raise RuntimeError("batch offsets missing range")
    start = int(offsets[0])
    end = int(offsets[1])
    if end <= start:
        raise RuntimeError("invalid first sample offsets in mixed batch")
    first = payload[start:end][0]
    if first == ord("A"):
        return "A"
    if first == ord("B"):
        return "B"
    raise RuntimeError(f"unexpected source tag byte={first}")


def _digest_ids(ids_sorted: list[str]) -> str:
    h = hashlib.sha256()
    for entry in ids_sorted:
        h.update(entry.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _rank_range(total: int, rank: int, world_size: int) -> tuple[int, int]:
    base = total // world_size
    rem = total % world_size
    start = rank * base + min(rank, rem)
    span = base + (1 if rank < rem else 0)
    end = start + span
    return start, end


def _worker(
    rank: int,
    world_size: int,
    total_samples: int,
    steps: int,
    seed: int,
    epoch: int,
    queue: mp.Queue,
) -> None:
    try:
        store_root = os.environ["MX8_MANIFEST_STORE_ROOT"]
        link_a = os.environ["MX8_DATASET_LINK_A"]
        link_b = os.environ["MX8_DATASET_LINK_B"]
        manifest_a = os.environ["MX8_DEV_MANIFEST_PATH_A"]
        manifest_b = os.environ["MX8_DEV_MANIFEST_PATH_B"]
        max_inflight_bytes = int(
            os.environ.get("MX8_MIX_GATE_MAX_INFLIGHT_BYTES", str(8 * 1024 * 1024))
        )
        max_process_rss_bytes = int(
            os.environ.get(
                "MX8_MIX_GATE_MAX_PROCESS_RSS_BYTES", str(2 * 1024 * 1024 * 1024)
            )
        )
        raw_weights = os.environ.get("MX8_MIX_GATE_WEIGHTS", "0.7,0.3")
        weights = [float(part.strip()) for part in raw_weights.split(",")]
        if len(weights) != 2 or any(w <= 0.0 for w in weights):
            raise RuntimeError(
                f"MX8_MIX_GATE_WEIGHTS must be two positive weights, got={weights}"
            )

        start_id, end_id = _rank_range(total_samples, rank, world_size)
        if end_id <= start_id:
            queue.put((rank, []))
            return

        loader_a = mx8.load(
            link_a,
            manifest_store=store_root,
            manifest_path=manifest_a,
            start_id=start_id,
            end_id=end_id,
            batch_size_samples=1,
            constraints=mx8.Constraints(
                max_inflight_bytes=max_inflight_bytes,
                max_ram_bytes=max_process_rss_bytes,
            ),
        )
        loader_b = mx8.load(
            link_b,
            manifest_store=store_root,
            manifest_path=manifest_b,
            start_id=start_id,
            end_id=end_id,
            batch_size_samples=1,
            constraints=mx8.Constraints(
                max_inflight_bytes=max_inflight_bytes,
                max_ram_bytes=max_process_rss_bytes,
            ),
        )

        mixed = mx8.mix(
            [loader_a, loader_b],
            weights=weights,
            seed=seed,
            epoch=epoch,
            source_exhausted="allow",
            profile="balanced",
            autotune=True,
            constraints=mx8.Constraints(
                max_inflight_bytes=max_inflight_bytes,
                max_ram_bytes=max_process_rss_bytes,
            ),
            runtime=mx8.RuntimeConfig(prefetch_batches=1, max_queue_batches=8),
        )

        ids: list[str] = []
        for _, batch in zip(range(steps), mixed):
            sample_ids = batch.sample_ids
            if len(sample_ids) != 1:
                raise RuntimeError(f"expected batch_size_samples=1, got={len(sample_ids)}")
            source = _source_tag(batch)
            ids.append(f"{source}:{int(sample_ids[0])}")
        queue.put((rank, ids))
    except Exception:
        traceback.print_exc()
        raise


def _run_once(
    world_size: int,
    total_samples: int,
    steps: int,
    seed: int,
    epoch: int,
) -> tuple[list[list[str]], list[str], int, int]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker,
            args=(rank, world_size, total_samples, steps, seed, epoch, queue),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()

    results: list[list[str] | None] = [None] * world_size
    pending = world_size
    while pending > 0:
        try:
            rank, ids = queue.get(timeout=1.0)
        except py_queue.Empty:
            failed = [proc for proc in procs if proc.exitcode not in (None, 0)]
            if failed:
                for proc in procs:
                    if proc.is_alive():
                        proc.terminate()
                for proc in procs:
                    proc.join(timeout=1.0)
                fail_desc = ", ".join(
                    f"pid={proc.pid} exit={proc.exitcode}" for proc in failed
                )
                raise RuntimeError(
                    f"mix multirank gate worker exited before publishing results: {fail_desc}"
                )
            if not any(proc.is_alive() for proc in procs):
                break
            continue

        rank_idx = int(rank)
        if rank_idx < 0 or rank_idx >= world_size:
            raise RuntimeError(f"invalid rank from worker: {rank_idx}")
        if results[rank_idx] is not None:
            raise RuntimeError(f"duplicate result from rank {rank_idx}")
        results[rank_idx] = list(ids)
        pending -= 1

    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(
                f"mix multirank gate failed: process {proc.pid} exit code {proc.exitcode}"
            )

    missing_ranks = [rank for rank, ids in enumerate(results) if ids is None]
    if missing_ranks:
        raise RuntimeError(
            f"missing worker results from ranks={missing_ranks}; "
            "workers may have exited before publishing to queue"
        )

    per_rank = [ids or [] for ids in results]
    total_ids = 0
    union: set[str] = set()
    digests: list[str] = []

    for rank, ids in enumerate(per_rank):
        total_ids += len(ids)
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicates within rank {rank}")
        # Preserve emission order for digest so epoch/seed schedule changes are observable.
        digests.append(_digest_ids(ids))
        overlap = union.intersection(ids)
        if overlap:
            first = sorted(overlap)[0]
            raise RuntimeError(
                f"overlap detected: sample={first} (rank={rank} and earlier rank)"
            )
        union.update(ids)

    return per_rank, digests, total_ids, len(union)


def _run_with_weights(
    world_size: int,
    total_samples: int,
    steps: int,
    seed: int,
    epoch: int,
    weights_csv: str | None,
) -> tuple[list[list[str]], list[str], int, int]:
    prev = os.environ.get("MX8_MIX_GATE_WEIGHTS")
    try:
        if weights_csv is None:
            os.environ.pop("MX8_MIX_GATE_WEIGHTS", None)
        else:
            os.environ["MX8_MIX_GATE_WEIGHTS"] = weights_csv
        return _run_once(
            world_size=world_size,
            total_samples=total_samples,
            steps=steps,
            seed=seed,
            epoch=epoch,
        )
    finally:
        if prev is None:
            os.environ.pop("MX8_MIX_GATE_WEIGHTS", None)
        else:
            os.environ["MX8_MIX_GATE_WEIGHTS"] = prev


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "4"))
    if world_size <= 0:
        raise RuntimeError(f"WORLD_SIZE must be >= 1, got={world_size}")
    total_samples = int(os.environ.get("MX8_TOTAL_SAMPLES", "2048"))
    steps = int(os.environ.get("MX8_MIX_DDP_STEPS", "128"))
    seed = int(os.environ.get("MX8_MIX_GATE_SEED", "17"))
    epoch = int(os.environ.get("MX8_MIX_GATE_EPOCH", "3"))
    expect_epoch_drift = (
        os.environ.get("MX8_MIX_GATE_EXPECT_EPOCH_DRIFT", "1").strip() == "1"
    )
    drift_probe_weights = os.environ.get("MX8_MIX_GATE_EPOCH_DRIFT_WEIGHTS", "1.0,1.0")

    per_rank_a, digests_a, total_ids_a, union_ids_a = _run_once(
        world_size=world_size,
        total_samples=total_samples,
        steps=steps,
        seed=seed,
        epoch=epoch,
    )
    _per_rank_b, digests_b, _total_ids_b, _union_ids_b = _run_once(
        world_size=world_size,
        total_samples=total_samples,
        steps=steps,
        seed=seed,
        epoch=epoch,
    )
    if digests_a != digests_b:
        raise RuntimeError(
            f"mix multirank replay mismatch for same epoch: {digests_a} vs {digests_b}"
        )

    _per_rank_c, digests_c, _total_ids_c, _union_ids_c = _run_once(
        world_size=world_size,
        total_samples=total_samples,
        steps=steps,
        seed=seed,
        epoch=epoch + 1,
    )
    if expect_epoch_drift and digests_a == digests_c:
        # Some imbalanced weights can produce the same finite prefix across epochs.
        # Probe with tie-heavy weights to verify epoch-dependent tie breaking.
        _probe_rank_a, probe_a, _probe_total_a, _probe_union_a = _run_with_weights(
            world_size=world_size,
            total_samples=total_samples,
            steps=steps,
            seed=seed,
            epoch=epoch,
            weights_csv=drift_probe_weights,
        )
        _probe_rank_b, probe_b, _probe_total_b, _probe_union_b = _run_with_weights(
            world_size=world_size,
            total_samples=total_samples,
            steps=steps,
            seed=seed,
            epoch=epoch + 1,
            weights_csv=drift_probe_weights,
        )
        if probe_a == probe_b:
            raise RuntimeError(
                "mix multirank epoch drift expected but digests unchanged for both "
                f"default and probe weights={drift_probe_weights!r}: {digests_a}"
            )

    print("ranks:", world_size)
    print("steps:", steps)
    print("total_ids:", total_ids_a)
    print("union_ids:", union_ids_a)
    print("dupes_within_rank:", 0)
    print("overlap_across_ranks:", 0)
    print("digests:", ",".join(digests_a))
    print("replay_determinism_same_epoch:", "ok")
    if expect_epoch_drift:
        print("epoch_drift_digest_change:", "ok")
    print("rank0_sequence_head:", ",".join(per_rank_a[0][:12]))


if __name__ == "__main__":
    main()
