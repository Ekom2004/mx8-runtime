import os

import mx8


def main() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"PyTorch is required for this example: {e}")

    root = os.environ.get("MX8_MANIFEST_STORE_ROOT", "/tmp/mx8-manifests")
    dev_manifest = os.environ.get("MX8_DEV_MANIFEST_PATH")
    if not dev_manifest:
        raise SystemExit("MX8_DEV_MANIFEST_PATH is required for the dev snapshot resolver")

    link = os.environ.get("MX8_DATASET_LINK", "/tmp/dev@refresh")
    loader = mx8.DataLoader(
        link,
        manifest_store=root,
        manifest_path=dev_manifest,
        batch_size_samples=8,
        max_inflight_bytes=64 * 1024,
        max_queue_batches=16,
        prefetch_batches=4,
    )

    total = 0
    for batch in loader:
        payload_u8, offsets_i64, sample_ids_i64 = batch.to_torch()
        assert payload_u8.dtype == __import__("torch").uint8
        assert offsets_i64.dtype == __import__("torch").int64
        assert sample_ids_i64.dtype == __import__("torch").int64

        # Ragged slicing: bytes for sample i are payload[offsets[i]:offsets[i+1]].
        for i in range(len(sample_ids_i64)):
            start = int(offsets_i64[i].item())
            end = int(offsets_i64[i + 1].item())
            _sample_bytes = payload_u8[start:end]
            total += 1

    print("delivered_samples:", total)


if __name__ == "__main__":
    main()

