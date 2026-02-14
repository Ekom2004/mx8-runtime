import os
import socket

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
    lr: float,
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
) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )

    loader = mx8.DistributedDataLoader(
        coord_url=coord_url,
        job_id=job_id,
        node_id=node_id,
        batch_size_samples=32,
        max_inflight_bytes=8 * 1024 * 1024,
        max_queue_batches=16,
        prefetch_batches=2,
        want=1,
        progress_interval_ms=200,
    )

    model = torch.nn.Linear(256, 1)
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=lr)

    it = iter(loader)

    last_loss = None
    for _ in range(steps):
        batch = next(it)
        payload_u8, offsets_i64, sample_ids_i64 = batch.to_torch()

        lengths = offsets_i64[1:] - offsets_i64[:-1]
        if not bool((lengths == 256).all().item()):
            raise SystemExit("ddp demo expects fixed 256-byte samples")

        bsz = int(sample_ids_i64.numel())
        x = payload_u8.reshape(bsz, 256).float() / 255.0
        y = (sample_ids_i64 % 2).float()

        pred = ddp(x).squeeze(-1)
        loss = F.mse_loss(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        last_loss = float(loss.detach().cpu().item())

    # Prove the process group is healthy and we can reduce values.
    t = torch.tensor([last_loss], dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if rank == 0:
        avg = float(t.item()) / float(world_size)
        print("world_size:", world_size)
        print("avg_last_loss:", avg)

    dist.destroy_process_group()


def main() -> None:
    import torch.multiprocessing as mp

    coord_url = os.environ.get("MX8_COORD_URL", "http://127.0.0.1:50051")
    job_id = os.environ.get("MX8_JOB_ID", "m5-ddp-demo")

    steps = int(os.environ.get("MX8_TORCH_STEPS", "8"))
    lr = float(os.environ.get("MX8_TORCH_LR", "0.01"))

    # If launched under torchrun, honor its rank/world_size; otherwise spawn locally.
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
        )
        return

    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    master_port = _free_port()
    init_method = f"tcp://127.0.0.1:{master_port}"

    mp.spawn(
        _spawn_entry,
        args=(world_size, init_method, coord_url, job_id, steps, lr),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
