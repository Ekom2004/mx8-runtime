import os

import mx8


def main() -> None:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"PyTorch is required for this example: {e}")

    root = os.environ.get("MX8_MANIFEST_STORE_ROOT", "/tmp/mx8-manifests")
    dev_manifest = os.environ.get("MX8_DEV_MANIFEST_PATH")
    if not dev_manifest:
        raise SystemExit("MX8_DEV_MANIFEST_PATH is required for the dev snapshot resolver")

    link = os.environ.get("MX8_DATASET_LINK", "/tmp/dev@refresh")

    steps = int(os.environ.get("MX8_TORCH_STEPS", "8"))
    lr = float(os.environ.get("MX8_TORCH_LR", "0.01"))

    loader = mx8.DataLoader(
        link,
        manifest_store_root=root,
        dev_manifest_path=dev_manifest,
        batch_size_samples=32,
        max_inflight_bytes=8 * 1024 * 1024,
        max_queue_batches=16,
        prefetch_batches=4,
    )

    model = None
    opt = None

    step = 0
    last_loss = None
    for batch in loader:
        payload_u8, offsets_i64, sample_ids_i64 = batch.to_torch()

        # v0 training demo: assume fixed-size samples so we can reshape the payload
        # into a dense [B, bytes_per_sample] tensor efficiently.
        lengths = offsets_i64[1:] - offsets_i64[:-1]
        bytes_per_sample = int(lengths[0].item())
        if not bool((lengths == bytes_per_sample).all().item()):
            raise SystemExit(
                "expected fixed-size samples for this demo; got variable-length batch"
            )

        bsz = int(sample_ids_i64.numel())
        x = payload_u8.reshape(bsz, bytes_per_sample).float() / 255.0

        # Dummy label: stable, deterministic, and cheap.
        y = (sample_ids_i64 % 2).float()

        if model is None:
            model = torch.nn.Linear(bytes_per_sample, 1)
            opt = torch.optim.SGD(model.parameters(), lr=lr)

        assert model is not None
        assert opt is not None

        pred = model(x).squeeze(-1)
        loss = F.mse_loss(pred, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        last_loss = float(loss.detach().cpu().item())
        step += 1
        if step >= steps:
            break

    print("steps:", step)
    print("last_loss:", last_loss)


if __name__ == "__main__":
    main()

