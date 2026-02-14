from __future__ import annotations

import hashlib
import os
import socket
import struct

import mx8


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def _digest_u64(ids_sorted: list[int]) -> str:
    h = hashlib.sha256()
    for v in ids_sorted:
        h.update(struct.pack("<Q", int(v)))
    return h.hexdigest()[:16]


def _spawn_entry(
    rank: int,
    world_size: int,
    init_method: str,
    coord_url: str,
    job_id: str,
    steps: int,
    want: int,
) -> None:
    _run_worker(
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        coord_url=coord_url,
        job_id=job_id,
        node_id=f"rank{rank}",
        steps=steps,
        want=want,
    )


def _run_worker(
    *,
    rank: int,
    world_size: int,
    init_method: str,
    coord_url: str,
    job_id: str,
    node_id: str,
    steps: int,
    want: int,
) -> None:
    import datetime
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(
            seconds=int(os.environ.get("MX8_TORCH_DIST_TIMEOUT_S", "30"))
        ),
    )

    loader = mx8.DistributedDataLoader(
        coord_url=coord_url,
        job_id=job_id,
        node_id=node_id,
        batch_size_samples=int(os.environ.get("MX8_TORCH_BATCH_SIZE_SAMPLES", "32")),
        max_inflight_bytes=int(os.environ.get("MX8_MAX_INFLIGHT_BYTES", str(8 * 1024 * 1024))),
        max_queue_batches=int(os.environ.get("MX8_MAX_QUEUE_BATCHES", "16")),
        prefetch_batches=int(os.environ.get("MX8_PREFETCH_BATCHES", "2")),
        want=want,
        progress_interval_ms=int(os.environ.get("MX8_PROGRESS_INTERVAL_MS", "200")),
    )

    seen_ids: list[int] = []
    it = iter(loader)
    try:
        for _ in range(steps):
            batch = next(it)
            _payload_u8, _offsets_i64, sample_ids_i64 = batch.to_torch()
            for v in sample_ids_i64.tolist():
                seen_ids.append(int(v))
    finally:
        loader.close()

    # Gather full ID lists so rank0 can detect duplicates within a rank.
    gathered: list[list[int] | None] = [None] * world_size
    dist.all_gather_object(gathered, seen_ids)

    if rank == 0:
        union: set[int] = set()
        per_rank_digests: list[str] = []
        total_ids = 0

        for r in range(world_size):
            ids = gathered[r]
            if ids is None:
                raise SystemExit("rank0 missing gathered ids")
            total_ids += len(ids)
            if len(ids) != len(set(ids)):
                raise SystemExit(f"duplicates within rank {r}")

            ids_sorted = sorted(set(ids))
            digest = _digest_u64(ids_sorted)
            per_rank_digests.append(digest)

            overlap = union.intersection(ids_sorted)
            if overlap:
                first = next(iter(overlap))
                raise SystemExit(f"overlap detected: sample_id={first} (rank={r} and earlier rank)")
            union.update(ids_sorted)

        print("ranks:", world_size)
        print("want:", want)
        print("steps:", steps)
        print("total_ids:", total_ids)
        print("union_ids:", len(union))
        print("dupes_within_rank:", 0)
        print("overlap_across_ranks:", 0)
        print("digests:", ",".join(per_rank_digests))

    dist.destroy_process_group()


def main() -> None:
    import torch.multiprocessing as mp

    coord_url = os.environ.get("MX8_COORD_URL", "http://127.0.0.1:50051")
    job_id = os.environ.get("MX8_JOB_ID", "m5-ddp-nodupes")

    steps = int(os.environ.get("MX8_TORCH_NODUPES_STEPS", "64"))
    want = int(os.environ.get("MX8_DEV_LEASE_WANT", "1"))

    rank_env = os.environ.get("RANK")
    world_env = os.environ.get("WORLD_SIZE")

    if rank_env is not None and world_env is not None:
        rank = int(rank_env)
        world_size = int(world_env)
        node_id = os.environ.get("MX8_NODE_ID", f"rank{rank}")
        init_method = os.environ.get("INIT_METHOD")
        if not init_method:
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", str(_free_port()))
            init_method = f"tcp://{master_addr}:{master_port}"
        _run_worker(
            rank=rank,
            world_size=world_size,
            init_method=init_method,
            coord_url=coord_url,
            job_id=job_id,
            node_id=node_id,
            steps=steps,
            want=want,
        )
        return

    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    master_port = _free_port()
    init_method = f"tcp://127.0.0.1:{master_port}"

    mp.spawn(
        _spawn_entry,
        args=(world_size, init_method, coord_url, job_id, steps, want),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
