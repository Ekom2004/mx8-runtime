from __future__ import annotations

import os
import socket
import tempfile

import mx8


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def _atomic_torch_save(obj, path: str) -> None:
    import torch

    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    # Torch's zipfile writer can reject some filenames; keep tmp names simple and suffixed.
    fd, tmp = tempfile.mkstemp(prefix="mx8-ckpt-", suffix=".pt", dir=d)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def _spawn_entry(
    rank: int,
    world_size: int,
    init_method: str,
    coord_url: str,
    job_id: str,
    steps: int,
    lr: float,
    load_ckpt_path: str | None,
    save_ckpt_path: str | None,
    save_ckpt_step: int,
    crash_rank: int | None,
    crash_after_step: int | None,
) -> None:
    _run_worker(
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        coord_url=coord_url,
        job_id=job_id,
        node_id=f"rank{rank}",
        steps=steps,
        lr=lr,
        load_ckpt_path=load_ckpt_path,
        save_ckpt_path=save_ckpt_path,
        save_ckpt_step=save_ckpt_step,
        crash_rank=crash_rank,
        crash_after_step=crash_after_step,
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
    lr: float,
    load_ckpt_path: str | None,
    save_ckpt_path: str | None,
    save_ckpt_step: int,
    crash_rank: int | None,
    crash_after_step: int | None,
) -> None:
    import datetime
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(
            seconds=int(os.environ.get("MX8_TORCH_DIST_TIMEOUT_S", "30"))
        ),
    )

    model = torch.nn.Linear(256, 1)
    if load_ckpt_path:
        ckpt = torch.load(load_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=lr)

    loader = mx8.DistributedDataLoader(
        coord_url=coord_url,
        job_id=job_id,
        node_id=node_id,
        batch_size_samples=int(os.environ.get("MX8_TORCH_BATCH_SIZE_SAMPLES", "32")),
        max_inflight_bytes=int(os.environ.get("MX8_MAX_INFLIGHT_BYTES", str(8 * 1024 * 1024))),
        max_queue_batches=int(os.environ.get("MX8_MAX_QUEUE_BATCHES", "16")),
        prefetch_batches=int(os.environ.get("MX8_PREFETCH_BATCHES", "2")),
        want=int(os.environ.get("MX8_DEV_LEASE_WANT", "1")),
        progress_interval_ms=int(os.environ.get("MX8_PROGRESS_INTERVAL_MS", "200")),
    )

    it = iter(loader)
    last_loss = None

    try:
        for step in range(1, steps + 1):
            batch = next(it)
            payload_u8, offsets_i64, sample_ids_i64 = batch.to_torch()

            lengths = offsets_i64[1:] - offsets_i64[:-1]
            if not bool((lengths == 256).all().item()):
                raise SystemExit("ckpt demo expects fixed 256-byte samples")

            bsz = int(sample_ids_i64.numel())
            x = payload_u8.reshape(bsz, 256).float() / 255.0
            y = (sample_ids_i64 % 2).float()

            pred = ddp(x).squeeze(-1)
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            last_loss = float(loss.detach().cpu().item())

            if save_ckpt_path and step == save_ckpt_step and rank == 0:
                _atomic_torch_save({"model": ddp.module.state_dict(), "step": step}, save_ckpt_path)
                print("saved_checkpoint:", save_ckpt_path)

            if crash_rank is not None and crash_after_step is not None:
                if rank == crash_rank and step == crash_after_step:
                    print("crashing_rank:", rank, "after_step:", step)
                    os._exit(42)
    finally:
        loader.close()

    # Prove we can communicate at the end (when not crashing).
    t = torch.tensor([last_loss], dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if rank == 0:
        avg = float(t.item()) / float(world_size)
        print("world_size:", world_size)
        print("avg_last_loss:", avg)
        print("loaded_checkpoint:", bool(load_ckpt_path))

    dist.destroy_process_group()


def main() -> None:
    import torch.multiprocessing as mp

    coord_url = os.environ.get("MX8_COORD_URL", "http://127.0.0.1:50051")
    job_id = os.environ.get("MX8_JOB_ID", "m5-ddp-ckpt-demo")

    steps = int(os.environ.get("MX8_TORCH_STEPS", "8"))
    lr = float(os.environ.get("MX8_TORCH_LR", "0.01"))

    load_ckpt_path = os.environ.get("MX8_TORCH_LOAD_CKPT_PATH") or None
    save_ckpt_path = os.environ.get("MX8_TORCH_SAVE_CKPT_PATH") or None
    save_ckpt_step = int(os.environ.get("MX8_TORCH_SAVE_CKPT_STEP", "2"))

    crash_rank = os.environ.get("MX8_TORCH_CRASH_RANK")
    crash_after_step = os.environ.get("MX8_TORCH_CRASH_AFTER_STEP")
    crash_rank_i = int(crash_rank) if crash_rank is not None else None
    crash_after_i = int(crash_after_step) if crash_after_step is not None else None

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
            lr=lr,
            load_ckpt_path=load_ckpt_path,
            save_ckpt_path=save_ckpt_path,
            save_ckpt_step=save_ckpt_step,
            crash_rank=crash_rank_i,
            crash_after_step=crash_after_i,
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
            lr,
            load_ckpt_path,
            save_ckpt_path,
            save_ckpt_step,
            crash_rank_i,
            crash_after_i,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
