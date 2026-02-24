import os

import mx8


def main() -> None:
    root = os.environ.get("MX8_MANIFEST_STORE_ROOT", "/tmp/mx8-manifests")
    dev_manifest = os.environ.get("MX8_DEV_MANIFEST_PATH")
    if not dev_manifest:
        raise SystemExit("MX8_DEV_MANIFEST_PATH is required for the dev snapshot resolver")

    link = os.environ.get("MX8_DATASET_LINK", "/tmp/dev@refresh")
    loader = mx8.load(
        link,
        manifest_store=root,
        manifest_path=dev_manifest,
        batch_size_samples=8,
        max_ram_gb=1,
        profile="safe",
    )

    iterator = iter(loader)
    try:
        next(iterator)
    except StopIteration:
        pass

    human = mx8.stats(loader)
    if not isinstance(human, str):
        raise AssertionError("mx8.stats(loader) must return a string")
    for required in ("Status:", "Mode:", "Progress:", "Throughput:", "Memory:", "Stability:"):
        if required not in human:
            raise AssertionError(f"mx8.stats output missing {required!r}\n{human}")
    if "inflight" in human.lower():
        raise AssertionError(f"mx8.stats output must not mention inflight\n{human}")

    raw = mx8.stats(loader, raw=True)
    direct = loader.stats()
    if set(raw.keys()) != set(direct.keys()):
        raise AssertionError("mx8.stats(loader, raw=True) must expose the same keys as loader.stats()")
    for stable_key in ("delivered_samples_total", "delivered_batches_total"):
        if raw.get(stable_key) != direct.get(stable_key):
            raise AssertionError(
                f"mx8.stats(loader, raw=True) mismatch for stable field {stable_key}"
            )

    for key in (
        "delivered_samples_total",
        "process_rss_bytes",
        "max_process_rss_bytes",
        "elapsed_seconds",
    ):
        if key not in raw:
            raise AssertionError(f"raw stats missing key: {key}")

    loader.close()
    print("[mx8] m5_stats_gate OK")


if __name__ == "__main__":
    main()
