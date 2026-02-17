import os
import time

import mx8


def run_once(
    backend: str,
    dataset_link: str,
    store_root: str,
    batch_size: int,
    steps: int,
    warmup_steps: int,
) -> tuple[int, int]:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"PyTorch is required for this benchmark: {e}")

    torch.manual_seed(0)
    torch.set_num_threads(1)
    os.environ["MX8_DECODE_BACKEND"] = backend

    def run_phase(target_steps: int) -> tuple[int, int]:
        loader = mx8.vision.ImageFolderLoader(
            dataset_link,
            manifest_store_root=store_root,
            batch_size_samples=batch_size,
            max_inflight_bytes=256 * 1024 * 1024,
            max_queue_batches=64,
            prefetch_batches=4,
            node_id=f"py_decode_bench_{backend}",
            to_float=True,
        )

        model = None
        opt = None
        step = 0
        sample_total = 0
        start = time.perf_counter()

        for x, ys in loader:
            x = x.contiguous()

            if model is None:
                c, h, w = x.shape[1], x.shape[2], x.shape[3]
                model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(c * h * w, 2))
                opt = torch.optim.SGD(model.parameters(), lr=0.05)

            logits = model(x)
            loss = F.cross_entropy(logits, ys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            sample_total += int(x.shape[0])
            step += 1
            if step >= target_steps:
                break

        if step < target_steps:
            raise SystemExit(
                f"backend={backend} requested_steps={target_steps} but only ran {step}; increase dataset size"
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return elapsed_ms, sample_total

    if warmup_steps > 0:
        _ = run_phase(warmup_steps)

    return run_phase(steps)


def main() -> None:
    dataset_link = os.environ.get("MX8_DATASET_LINK")
    if not dataset_link:
        raise SystemExit("MX8_DATASET_LINK is required")

    store_root = os.environ.get("MX8_MANIFEST_STORE_ROOT", "/tmp/mx8-manifests")
    batch_size = int(os.environ.get("MX8_VISION_BENCH_BATCH_SIZE", "64"))
    steps = int(os.environ.get("MX8_VISION_BENCH_STEPS", "128"))
    warmup_steps = int(os.environ.get("MX8_VISION_BENCH_WARMUP_STEPS", "8"))
    min_speedup = float(os.environ.get("MX8_DECODE_BENCH_MIN_SPEEDUP", "0"))

    rust_elapsed_ms, rust_samples = run_once(
        "rust", dataset_link, store_root, batch_size, steps, warmup_steps
    )
    py_elapsed_ms, py_samples = run_once(
        "python", dataset_link, store_root, batch_size, steps, warmup_steps
    )

    if rust_samples != py_samples:
        raise SystemExit(
            f"sample mismatch: rust_samples={rust_samples} python_samples={py_samples}"
        )

    rust_sps = rust_samples / (rust_elapsed_ms / 1000.0)
    py_sps = py_samples / (py_elapsed_ms / 1000.0)
    speedup = py_elapsed_ms / rust_elapsed_ms

    print(
        f"backend=rust steps={steps} samples={rust_samples} elapsed_ms={rust_elapsed_ms} samples_per_sec={rust_sps:.3f}"
    )
    print(
        f"backend=python steps={steps} samples={py_samples} elapsed_ms={py_elapsed_ms} samples_per_sec={py_sps:.3f}"
    )
    print(f"decode_speedup_rust_over_python={speedup:.4f}x")

    if min_speedup > 0 and speedup < min_speedup:
        raise SystemExit(
            f"decode speedup gate failed: speedup={speedup:.4f} < min_speedup={min_speedup:.4f}"
        )


if __name__ == "__main__":
    main()
