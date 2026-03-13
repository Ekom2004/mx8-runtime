#!/usr/bin/env python3
import os
from typing import Any, Dict, List, Tuple

import mx8


def ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def loader_kwargs() -> Dict[str, Any]:
    data = os.environ["MX8_DATASET_LINK"]
    kwargs: Dict[str, Any] = {
        "data": data,
        "batch": 8,
        "ram_gb": 1.0,
        "profile": "safe",
        "prefetch": 1,
        "queue": 4,
        "inflight": 32 * 1024 * 1024,
    }
    store = os.environ.get("MX8_MANIFEST_STORE_ROOT")
    manifest = os.environ.get("MX8_DEV_MANIFEST_PATH")
    if store:
        kwargs["store"] = store
    if manifest:
        kwargs["manifest"] = manifest
    return kwargs


def next_batch_with_transform(transform_fn) -> Tuple[bytes, List[int], List[int], Dict[str, Any]]:
    kwargs = loader_kwargs()
    loader = mx8.load(transform=transform_fn, **kwargs)
    try:
        batch = next(iter(loader))
        stats_raw = loader.stats()
        return bytes(batch.payload), list(batch.offsets), list(batch.sample_ids), dict(stats_raw)
    finally:
        loader.close()


@mx8.transform
def tag_prefix(sample: bytes) -> bytes:
    return b"T" + sample


@mx8.transform
def duplicate(sample: bytes) -> bytes:
    return sample + sample


def every_sample_prefixed(payload: bytes, offsets: List[int], prefix: bytes) -> bool:
    for i in range(len(offsets) - 1):
        start = offsets[i]
        end = offsets[i + 1]
        if end <= start:
            continue
        if payload[start : start + len(prefix)] != prefix:
            return False
    return True


def expect_undecorated_rejected() -> None:
    kwargs = loader_kwargs()

    def plain(sample: bytes) -> bytes:
        return sample

    got_error = False
    try:
        _ = mx8.load(transform=plain, **kwargs)
    except Exception:
        got_error = True
    ensure(got_error, "undecorated callable should be rejected by transform fail-fast validation")


def expect_cap_violation() -> None:
    prev = os.environ.get("MX8_TRANSFORM_MAX_OUTPUT_BYTES_PER_SAMPLE")
    os.environ["MX8_TRANSFORM_MAX_OUTPUT_BYTES_PER_SAMPLE"] = "8"
    try:
        kwargs = loader_kwargs()
        loader = mx8.load(transform=duplicate, **kwargs)
        try:
            got_error = False
            try:
                _ = next(iter(loader))
            except Exception:
                got_error = True
            ensure(got_error, "transform output cap breach should fail the loader")
        finally:
            loader.close()
    finally:
        if prev is None:
            os.environ.pop("MX8_TRANSFORM_MAX_OUTPUT_BYTES_PER_SAMPLE", None)
        else:
            os.environ["MX8_TRANSFORM_MAX_OUTPUT_BYTES_PER_SAMPLE"] = prev


def main() -> None:
    payload_a, offsets_a, ids_a, stats_a = next_batch_with_transform(tag_prefix)
    payload_b, offsets_b, ids_b, stats_b = next_batch_with_transform(tag_prefix)

    ensure(payload_a == payload_b, "determinism failure: payload mismatch across identical runs")
    ensure(offsets_a == offsets_b, "determinism failure: offsets mismatch across identical runs")
    ensure(ids_a == ids_b, "determinism failure: sample_ids mismatch across identical runs")
    ensure(
        every_sample_prefixed(payload_a, offsets_a, b"T"),
        "startup cutover failure: observed untransformed samples in first delivered batch",
    )
    ensure(stats_a.get("transform_enabled") is True, "expected transform_enabled=True in loader stats")
    ensure(stats_a.get("transform_samples_total", 0) > 0, "expected transform_samples_total > 0")
    ensure(
        stats_b.get("transform_output_ratio_ewma", 0.0) >= 1.0,
        "expected transform_output_ratio_ewma to reflect transformed output",
    )

    expect_undecorated_rejected()
    expect_cap_violation()

    print("[mx8] m9_transform_gate OK")


if __name__ == "__main__":
    main()
