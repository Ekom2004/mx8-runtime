#!/usr/bin/env python3
import os

import mx8
import torch


def next_batch(dataset_link: str, manifest_store: str, **kwargs):
    loader = mx8.image(
        dataset_link,
        manifest_store=manifest_store,
        batch_size_samples=2,
        prefetch_batches=2,
        max_queue_batches=8,
        max_inflight_bytes=64 * 1024 * 1024,
        **kwargs,
    )
    try:
        return next(iter(loader))
    finally:
        loader.close()


def ensure(cond: bool, msg: str):
    if not cond:
        raise SystemExit(msg)


def main():
    dataset_link = os.environ["MX8_DATASET_LINK"]
    manifest_store = os.environ["MX8_MANIFEST_STORE_ROOT"]

    # Preset determinism: same seed+epoch+snapshot => identical first batch.
    a_images, a_labels = next_batch(
        dataset_link,
        manifest_store,
        resize_hw=(256, 256),
        augment="imagenet",
        seed=17,
        epoch=3,
    )
    b_images, b_labels = next_batch(
        dataset_link,
        manifest_store,
        resize_hw=(256, 256),
        augment="imagenet",
        seed=17,
        epoch=3,
    )
    ensure(torch.equal(a_images, b_images), "expected deterministic images for same seed+epoch")
    ensure(torch.equal(a_labels, b_labels), "expected deterministic labels for same seed+epoch")
    ensure(tuple(a_images.shape) == (2, 3, 224, 224), f"unexpected imagenet shape: {tuple(a_images.shape)}")
    ensure(a_images.dtype == torch.float32, f"expected float32, got {a_images.dtype}")
    ensure(a_labels.dtype == torch.int64, f"expected int64 labels, got {a_labels.dtype}")

    # Epoch perturbation should change augmentation choices.
    c_images, c_labels = next_batch(
        dataset_link,
        manifest_store,
        resize_hw=(256, 256),
        augment="imagenet",
        seed=17,
        epoch=4,
    )
    ensure(torch.equal(a_labels, c_labels), "labels should not change across epochs")
    ensure(
        not torch.equal(a_images, c_images),
        "expected different images when epoch changes",
    )

    # Explicit knobs path.
    d_images, d_labels = next_batch(
        dataset_link,
        manifest_store,
        resize_hw=(256, 256),
        crop_hw=(224, 224),
        horizontal_flip_p=0.5,
        color_jitter_brightness=0.2,
        color_jitter_contrast=0.2,
        color_jitter_saturation=0.2,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        seed=9,
        epoch=1,
    )
    ensure(tuple(d_images.shape) == (2, 3, 224, 224), f"unexpected explicit-knob shape: {tuple(d_images.shape)}")
    ensure(d_images.dtype == torch.float32, f"expected float32, got {d_images.dtype}")
    ensure(d_labels.dtype == torch.int64, f"expected int64 labels, got {d_labels.dtype}")

    # to_float=False + geometric-only augmentation keeps uint8 output.
    e_images, _ = next_batch(
        dataset_link,
        manifest_store,
        resize_hw=(256, 256),
        crop_hw=(224, 224),
        horizontal_flip_p=0.5,
        seed=5,
        epoch=2,
        to_float=False,
    )
    ensure(e_images.dtype == torch.uint8, f"expected uint8 output for to_float=False, got {e_images.dtype}")

    print("[mx8] m6_image_aug_gate OK")


if __name__ == "__main__":
    main()
