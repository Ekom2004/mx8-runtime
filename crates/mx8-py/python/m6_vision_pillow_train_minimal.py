import os

import mx8


def main() -> None:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"PyTorch is required for this example: {e}")

    root = os.environ.get("MX8_MANIFEST_STORE_ROOT", "/tmp/mx8-manifests")
    link = os.environ.get("MX8_DATASET_LINK")
    if not link:
        raise SystemExit("MX8_DATASET_LINK is required (e.g. s3://bucket/prefix@refresh)")

    batch_size = int(os.environ.get("MX8_BATCH_SIZE_SAMPLES", "4"))
    steps = int(os.environ.get("MX8_TRAIN_STEPS", "8"))

    loader = mx8.vision.ImageFolderLoader(
        link,
        manifest_store_root=root,
        batch_size_samples=batch_size,
        max_inflight_bytes=128 * 1024 * 1024,
        max_queue_batches=32,
        prefetch_batches=4,
        node_id=os.environ.get("MX8_NODE_ID", "py_train"),
        to_float=True,
    )

    model = None
    opt = None

    step = 0
    last_loss = None
    for x, ys in loader:
        x = x.contiguous()

        if model is None:
            c, h, w = x.shape[1], x.shape[2], x.shape[3]
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(c * h * w, 2))
            opt = torch.optim.SGD(model.parameters(), lr=0.1)

        logits = model(x)
        loss = F.cross_entropy(logits, ys)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        last_loss = float(loss.detach().cpu().item())
        step += 1
        if step >= steps:
            break

    if last_loss is None:
        raise SystemExit("no steps executed")

    print("steps:", step)
    print("last_loss:", last_loss)


if __name__ == "__main__":
    main()
