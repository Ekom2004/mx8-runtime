import os
import time
from typing import Optional

import mx8


def parse_csv(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    out = [part.strip() for part in raw.split(",") if part.strip()]
    if not out:
        raise SystemExit(f"{name} produced an empty list")
    return out


def parse_int_csv(name: str, default: str) -> list[int]:
    out: list[int] = []
    for token in parse_csv(name, default):
        try:
            value = int(token)
        except ValueError as e:
            raise SystemExit(f"invalid integer in {name}: {token!r}") from e
        if value < 1:
            raise SystemExit(f"{name} values must be >= 1 (got {value})")
        out.append(value)
    return out


def parse_rust_codecs(name: str, default: str) -> list[str]:
    codecs = parse_csv(name, default)
    allowed = {"zune", "image", "turbo"}
    out: list[str] = []
    for codec in codecs:
        if codec not in allowed:
            raise SystemExit(
                f"unsupported rust codec {codec!r} in {name} (allowed: {sorted(allowed)})"
            )
        out.append(codec)
    return out


def run_once(
    backend: str,
    rust_codec: Optional[str],
    dataset_link: str,
    store_root: str,
    batch_size: int,
    steps: int,
    warmup_steps: int,
    mode: str,
    torch_threads: int,
) -> tuple[int, int]:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"PyTorch is required for this benchmark: {e}")

    torch.manual_seed(0)
    torch.set_num_threads(torch_threads)
    os.environ["MX8_DECODE_BACKEND"] = backend
    if rust_codec is not None:
        os.environ["MX8_RUST_JPEG_CODEC"] = rust_codec
    else:
        os.environ.pop("MX8_RUST_JPEG_CODEC", None)

    def run_phase(target_steps: int) -> tuple[int, int]:
        loader = mx8.image(
            dataset_link,
            manifest_store=store_root,
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

            if mode == "train_step":
                if model is None:
                    c, h, w = x.shape[1], x.shape[2], x.shape[3]
                    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(c * h * w, 2))
                    opt = torch.optim.SGD(model.parameters(), lr=0.05)

                logits = model(x)
                loss = F.cross_entropy(logits, ys)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            elif mode == "decode_only":
                _ = ys.shape[0]
            else:  # pragma: no cover
                raise SystemExit(f"unsupported mode: {mode}")

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
    backends = parse_csv("MX8_VISION_BENCH_BACKENDS", "rust,python")
    rust_codecs = parse_rust_codecs("MX8_VISION_BENCH_RUST_CODECS", "zune,image")
    modes = parse_csv("MX8_VISION_BENCH_MODES", "decode_only,train_step")
    torch_threads_list = parse_int_csv("MX8_VISION_BENCH_TORCH_THREADS_LIST", "1")
    decode_threads_list = parse_int_csv("MX8_VISION_BENCH_DECODE_THREADS_LIST", "1")
    batch_size = int(os.environ.get("MX8_VISION_BENCH_BATCH_SIZE", "64"))
    steps = int(os.environ.get("MX8_VISION_BENCH_STEPS", "128"))
    warmup_steps = int(os.environ.get("MX8_VISION_BENCH_WARMUP_STEPS", "8"))
    min_speedup = float(os.environ.get("MX8_DECODE_BENCH_MIN_SPEEDUP", "0"))

    if batch_size < 1 or steps < 1 or warmup_steps < 0:
        raise SystemExit(
            "invalid benchmark config: batch_size>=1, steps>=1, warmup_steps>=0 required"
        )

    known_modes = {"decode_only", "train_step"}
    for mode in modes:
        if mode not in known_modes:
            raise SystemExit(
                f"unsupported mode {mode!r} in MX8_VISION_BENCH_MODES (allowed: {sorted(known_modes)})"
            )

    results: dict[tuple[str, int, int, str, str], tuple[int, int]] = {}

    for mode in modes:
        for torch_threads in torch_threads_list:
            for decode_threads in decode_threads_list:
                os.environ["MX8_DECODE_THREADS"] = str(decode_threads)
                for backend in backends:
                    backend_codecs = rust_codecs if backend == "rust" else ["na"]
                    for rust_codec in backend_codecs:
                        elapsed_ms, samples = run_once(
                            backend=backend,
                            rust_codec=None if rust_codec == "na" else rust_codec,
                            dataset_link=dataset_link,
                            store_root=store_root,
                            batch_size=batch_size,
                            steps=steps,
                            warmup_steps=warmup_steps,
                            mode=mode,
                            torch_threads=torch_threads,
                        )
                        results[
                            (mode, torch_threads, decode_threads, backend, rust_codec)
                        ] = (
                            elapsed_ms,
                            samples,
                        )
                        samples_per_sec = samples / (elapsed_ms / 1000.0)
                        print(
                            "mode="
                            f"{mode} torch_threads={torch_threads} decode_threads={decode_threads} "
                            f"backend={backend} rust_codec={rust_codec} "
                            f"steps={steps} samples={samples} elapsed_ms={elapsed_ms} "
                            f"samples_per_sec={samples_per_sec:.3f}"
                        )

    compares: list[tuple[str, int, int, str, float]] = []
    for mode in modes:
        for torch_threads in torch_threads_list:
            for decode_threads in decode_threads_list:
                py_key = (mode, torch_threads, decode_threads, "python", "na")
                if py_key not in results:
                    continue

                py_elapsed_ms, py_samples = results[py_key]
                for rust_codec in rust_codecs:
                    rust_key = (mode, torch_threads, decode_threads, "rust", rust_codec)
                    if rust_key not in results:
                        continue

                    rust_elapsed_ms, rust_samples = results[rust_key]
                    if rust_samples != py_samples:
                        raise SystemExit(
                            "sample mismatch for compare "
                            f"(mode={mode} torch_threads={torch_threads} decode_threads={decode_threads} "
                            f"rust_codec={rust_codec}): "
                            f"rust_samples={rust_samples} python_samples={py_samples}"
                        )
                    speedup = py_elapsed_ms / rust_elapsed_ms
                    compares.append((mode, torch_threads, decode_threads, rust_codec, speedup))
                    print(
                        f"compare mode={mode} torch_threads={torch_threads} decode_threads={decode_threads} "
                        f"rust_codec={rust_codec} rust_over_python={speedup:.4f}x"
                    )
                    if min_speedup > 0 and speedup < min_speedup:
                        raise SystemExit(
                            "decode speedup gate failed: "
                            f"mode={mode} torch_threads={torch_threads} decode_threads={decode_threads} "
                            f"rust_codec={rust_codec} "
                            f"speedup={speedup:.4f} < min_speedup={min_speedup:.4f}"
                        )

    if len(compares) == 1:
        _, _, _, rust_codec, speedup = compares[0]
        print(f"decode_speedup_rust_over_python[{rust_codec}]={speedup:.4f}x")


if __name__ == "__main__":
    main()
