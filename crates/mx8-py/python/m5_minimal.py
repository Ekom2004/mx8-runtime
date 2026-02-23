import os

import mx8


def main() -> None:
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
        max_ram_gb=1,
        profile="safe",
    )

    print("manifest_hash:", loader.manifest_hash)

    total = 0
    for batch in loader:
        total += len(batch.sample_ids)
    print("delivered_samples:", total)


if __name__ == "__main__":
    main()

