#!/usr/bin/env python3
import os

import mx8
import torch


def ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def read_first_batch(dataset_link: str, manifest_store: str, policy: str):
    loader = mx8.audio(
        dataset_link,
        manifest_store=manifest_store,
        batch_size_samples=16,
        sample_count=4096,
        channels=1,
        sample_rate_hz=16000,
        decode_error_policy=policy,
        prefetch_batches=2,
        max_queue_batches=8,
        max_inflight_bytes=64 * 1024 * 1024,
    )
    try:
        batch = next(iter(loader))
        return batch.samples, batch.sample_rates_hz, batch.sample_ids
    finally:
        loader.close()


def main() -> None:
    dataset_link = os.environ["MX8_DATASET_LINK"]
    manifest_store = os.environ["MX8_MANIFEST_STORE_ROOT"]

    a_samples, a_rates, a_ids = read_first_batch(dataset_link, manifest_store, "skip")
    b_samples, b_rates, b_ids = read_first_batch(dataset_link, manifest_store, "skip")

    ensure(torch.equal(a_samples, b_samples), "expected deterministic audio samples for same input")
    ensure(torch.equal(a_rates, b_rates), "expected deterministic sample_rates_hz for same input")
    ensure(torch.equal(a_ids, b_ids), "expected deterministic sample_ids for same input")

    ensure(a_samples.dtype == torch.float32, f"expected float32 samples, got {a_samples.dtype}")
    ensure(a_rates.dtype == torch.int64, f"expected int64 sample rates, got {a_rates.dtype}")
    ensure(a_ids.dtype == torch.int64, f"expected int64 sample ids, got {a_ids.dtype}")
    ensure(tuple(a_samples.shape) == (2, 4096), f"unexpected samples shape: {tuple(a_samples.shape)}")
    ensure(tuple(a_rates.shape) == (2,), f"unexpected sample_rates_hz shape: {tuple(a_rates.shape)}")
    ensure(tuple(a_ids.shape) == (2,), f"unexpected sample_ids shape: {tuple(a_ids.shape)}")
    ensure(torch.all(a_rates == 16000).item(), f"expected sample rates of 16000, got {a_rates.tolist()}")
    ensure(torch.max(torch.abs(a_samples)).item() <= 1.001, "expected audio samples in [-1, 1] range")

    loader_err = mx8.audio(
        dataset_link,
        manifest_store=manifest_store,
        batch_size_samples=16,
        sample_count=4096,
        channels=1,
        sample_rate_hz=16000,
        decode_error_policy="error",
        prefetch_batches=2,
        max_queue_batches=8,
        max_inflight_bytes=64 * 1024 * 1024,
    )
    try:
        got_error = False
        try:
            _ = next(iter(loader_err))
        except Exception:
            got_error = True
        ensure(got_error, "decode_error_policy='error' should fail on invalid audio sample")
    finally:
        loader_err.close()

    print("[mx8] m6_audio_gate OK")


if __name__ == "__main__":
    main()
