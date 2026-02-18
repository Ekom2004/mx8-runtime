from __future__ import annotations

import os
import socket
import time
from typing import Dict, List, Optional

import mx8


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def _spawn_entry(
    rank: int,
    world_size: int,
    init_method: str,
    coord_url: str,
    job_id: str,
    steps: int,
    want: int,
    prefetch_batches: int,
    max_queue_batches: int,
    max_inflight_bytes: int,
    progress_interval_ms: int,
    compute_ms: float,
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
        prefetch_batches=prefetch_batches,
        max_queue_batches=max_queue_batches,
        max_inflight_bytes=max_inflight_bytes,
        progress_interval_ms=progress_interval_ms,
        compute_ms=compute_ms,
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
    prefetch_batches: int,
    max_queue_batches: int,
    max_inflight_bytes: int,
    progress_interval_ms: int,
    compute_ms: float,
) -> None:
    import datetime
    import torch
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
        batch_size_samples=int(os.environ.get("MX8_TORCH_BATCH_SIZE_SAMPLES", "64")),
        max_inflight_bytes=max_inflight_bytes,
        max_queue_batches=max_queue_batches,
        prefetch_batches=prefetch_batches,
        want=want,
        progress_interval_ms=progress_interval_ms,
    )

    wait_s_total = 0.0
    step_s_total = 0.0
    delivered_samples = 0
    it = iter(loader)
    try:
        for _ in range(steps):
            started = time.perf_counter()
            batch = next(it)
            fetched = time.perf_counter()
            payload_u8, offsets_i64, sample_ids_i64 = batch.to_torch()
            lengths = offsets_i64[1:] - offsets_i64[:-1]
            if not bool((lengths > 0).all().item()):
                raise SystemExit("non-positive sample length observed")
            _ = torch.sum(payload_u8[:16]).item()
            if compute_ms > 0:
                time.sleep(compute_ms / 1000.0)
            finished = time.perf_counter()

            wait_s_total += fetched - started
            step_s_total += finished - started
            delivered_samples += int(sample_ids_i64.numel())
    finally:
        stats = loader.stats()
        loader.close()

    rank_report = {
        "wait_s_total": wait_s_total,
        "step_s_total": step_s_total,
        "delivered_samples": delivered_samples,
        "process_rss_bytes": int(stats.get("process_rss_bytes", 0)),
        "ram_high_water_bytes": int(stats.get("ram_high_water_bytes", 0)),
        "autotune_enabled": bool(stats.get("autotune_enabled", False)),
        "effective_want": int(stats.get("effective_want", want)),
    }

    gathered: List[Optional[Dict[str, object]]] = [None] * world_size
    dist.all_gather_object(gathered, rank_report)

    if rank == 0:
        total_wait = 0.0
        total_step = 0.0
        total_samples = 0
        cap = int(os.environ.get("MX8_MAX_PROCESS_RSS_BYTES", "0"))
        cap_breach_ranks = 0
        max_process_rss = 0
        max_ram_high_water = 0
        autotune_enabled_ranks = 0
        wants = []

        for idx in range(world_size):
            report = gathered[idx]
            if report is None:
                raise SystemExit("missing rank report")
            total_wait += float(report["wait_s_total"])
            total_step += float(report["step_s_total"])
            total_samples += int(report["delivered_samples"])
            process_rss = int(report["process_rss_bytes"])
            ram_high_water = int(report["ram_high_water_bytes"])
            max_process_rss = max(max_process_rss, process_rss)
            max_ram_high_water = max(max_ram_high_water, ram_high_water)
            if cap > 0 and process_rss > cap:
                cap_breach_ranks += 1
            if bool(report["autotune_enabled"]):
                autotune_enabled_ranks += 1
            wants.append(int(report["effective_want"]))

        wait_ratio = (total_wait / total_step) if total_step > 0 else 0.0
        samples_per_sec = (total_samples / total_step) if total_step > 0 else 0.0
        effective_want_avg = sum(wants) / len(wants) if wants else 0.0

        print("mode:", os.environ.get("MX8_AUTOTUNE_AB_MODE", "unknown"))
        print("ranks:", world_size)
        print("steps:", steps)
        print("compute_ms:", f"{compute_ms:.3f}")
        print("wait_ratio:", f"{wait_ratio:.6f}")
        print("samples_per_sec:", f"{samples_per_sec:.3f}")
        print("total_samples:", total_samples)
        print("autotune_enabled_ranks:", autotune_enabled_ranks)
        print("effective_want_avg:", f"{effective_want_avg:.3f}")
        print("max_process_rss_bytes:", max_process_rss)
        print("max_ram_high_water_bytes:", max_ram_high_water)
        print("max_process_rss_cap_bytes:", cap)
        print("cap_breach_ranks:", cap_breach_ranks)

        if cap_breach_ranks > 0:
            raise SystemExit(
                f"rss cap breach detected: cap={cap} cap_breach_ranks={cap_breach_ranks}"
            )

    dist.destroy_process_group()


def main() -> None:
    import torch.multiprocessing as mp

    coord_url = os.environ.get("MX8_COORD_URL", "http://127.0.0.1:50051")
    job_id = os.environ.get("MX8_JOB_ID", "m5-ddp-autotune-ab")

    steps = int(os.environ.get("MX8_TORCH_AB_STEPS", "96"))
    want = int(os.environ.get("MX8_DEV_LEASE_WANT", "1"))
    prefetch_batches = int(os.environ.get("MX8_PREFETCH_BATCHES", "1"))
    max_queue_batches = int(os.environ.get("MX8_MAX_QUEUE_BATCHES", "8"))
    max_inflight_bytes = int(os.environ.get("MX8_MAX_INFLIGHT_BYTES", str(64 * 1024 * 1024)))
    progress_interval_ms = int(os.environ.get("MX8_PROGRESS_INTERVAL_MS", "200"))
    compute_ms = float(os.environ.get("MX8_TORCH_AB_COMPUTE_MS", "5"))

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
            prefetch_batches=prefetch_batches,
            max_queue_batches=max_queue_batches,
            max_inflight_bytes=max_inflight_bytes,
            progress_interval_ms=progress_interval_ms,
            compute_ms=compute_ms,
        )
        return

    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    master_port = _free_port()
    init_method = f"tcp://127.0.0.1:{master_port}"

    mp.spawn(
        _spawn_entry,
        args=(
            world_size,
            init_method,
            coord_url,
            job_id,
            steps,
            want,
            prefetch_batches,
            max_queue_batches,
            max_inflight_bytes,
            progress_interval_ms,
            compute_ms,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
