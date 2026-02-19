import os
from pathlib import Path

import mx8


def _make_local_videos(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Deterministic local payloads; .mp4 extension triggers stage1 video indexing.
    (root / "clip_a.mp4").write_bytes(b"A" * 480_000)
    (root / "clip_b.mp4").write_bytes(b"B" * 360_000)
    nested = root / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "clip_c.mp4").write_bytes(b"C" * 300_000)


def _collect_sequence(loader, max_batches: int) -> list[str]:
    seq = []
    for _idx, batch in zip(range(max_batches), loader):
        clip_ids = list(batch.clip_ids)
        if not clip_ids:
            raise RuntimeError("video batch missing clip_ids")
        seq.extend(str(x) for x in clip_ids)
        offsets = list(batch.offsets)
        payload = batch.payload
        if len(offsets) < 2:
            raise RuntimeError("video batch offsets malformed")
        if int(offsets[-1]) != len(payload):
            raise RuntimeError(
                f"video batch offsets/payload mismatch: tail={offsets[-1]} payload={len(payload)}"
            )
    return seq


def main() -> None:
    tmp_root = Path(os.environ["MX8_VIDEO_STAGE2A_TMP_ROOT"])
    store_root = tmp_root / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    ds_root = tmp_root / "videos"
    _make_local_videos(ds_root)

    # Stage2a CPU gate path uses stage1 local metadata extraction.
    os.environ["MX8_VIDEO_STAGE1_INDEX"] = "1"
    os.environ["MX8_VIDEO_STAGE1_DISABLE_FFPROBE"] = "1"
    os.environ["MX8_VIDEO_STAGE2_BYTES_PER_CLIP"] = os.environ.get(
        "MX8_VIDEO_STAGE2_BYTES_PER_CLIP", "1024"
    )

    clip_len = int(os.environ.get("MX8_VIDEO_STAGE2A_CLIP_LEN", "4"))
    stride = int(os.environ.get("MX8_VIDEO_STAGE2A_STRIDE", "2"))
    fps = int(os.environ.get("MX8_VIDEO_STAGE2A_FPS", "8"))
    batch_size = int(os.environ.get("MX8_VIDEO_STAGE2A_BATCH_SIZE", "4"))
    max_batches = int(os.environ.get("MX8_VIDEO_STAGE2A_MAX_BATCHES", "12"))
    seed = int(os.environ.get("MX8_VIDEO_STAGE2A_SEED", "23"))
    epoch = int(os.environ.get("MX8_VIDEO_STAGE2A_EPOCH", "5"))

    constraints = mx8.Constraints(max_inflight_bytes=64 * 1024, max_process_rss_bytes=None)

    loader1 = mx8.video(
        str(ds_root),
        manifest_store_root=str(store_root),
        recursive=True,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size_samples=batch_size,
        seed=seed,
        epoch=epoch,
        constraints=constraints,
    )
    seq1 = _collect_sequence(loader1, max_batches=max_batches)
    stats1 = loader1.stats()

    loader2 = mx8.video(
        str(ds_root),
        manifest_store_root=str(store_root),
        recursive=True,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size_samples=batch_size,
        seed=seed,
        epoch=epoch,
        constraints=constraints,
    )
    seq2 = _collect_sequence(loader2, max_batches=max_batches)
    if seq1 != seq2:
        raise RuntimeError("video stage2a replay mismatch for same seed/epoch")

    if int(stats1.get("video_delivered_batches_total", 0)) <= 0:
        raise RuntimeError(f"unexpected stats: {stats1}")
    if int(stats1.get("video_delivered_samples_total", 0)) <= 0:
        raise RuntimeError(f"unexpected stats: {stats1}")
    if int(stats1.get("video_delivered_bytes_total", 0)) <= 0:
        raise RuntimeError(f"unexpected stats: {stats1}")

    # Memory-bound init gate: make cap too small for configured batch.
    try:
        mx8.video(
            str(ds_root),
            manifest_store_root=str(store_root),
            recursive=True,
            clip_len=clip_len,
            stride=stride,
            fps=fps,
            batch_size_samples=batch_size,
            seed=seed,
            epoch=epoch,
            constraints=mx8.Constraints(max_inflight_bytes=2 * 1024, max_process_rss_bytes=None),
        )
        raise RuntimeError("expected max_inflight_bytes guard to reject config")
    except ValueError as e:
        if "exceeds max_inflight_bytes" not in str(e):
            raise RuntimeError(f"unexpected config error: {e}")

    print("video_stage2a_gate_summary")
    print("  batches:", int(stats1["video_delivered_batches_total"]))
    print("  samples:", int(stats1["video_delivered_samples_total"]))
    print("  bytes:", int(stats1["video_delivered_bytes_total"]))
    print("  clips_total:", int(stats1["clips_total"]))
    print("  replay_determinism: ok")
    print("  memory_bound_init_gate: ok")


if __name__ == "__main__":
    main()
