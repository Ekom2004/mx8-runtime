import os
from pathlib import Path

import mx8


def _write_bytes(path: Path, fill: bytes, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fill * n)


def _make_local_videos(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_bytes(root / "clip_a.mp4", b"A", 480_000)
    _write_bytes(root / "clip_b.mp4", b"B", 360_000)
    _write_bytes(root / "nested" / "clip_c.mp4", b"C", 300_000)


def _make_failure_probe_video(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "missing_soon.mp4"
    _write_bytes(target, b"Z", 256_000)
    return target


def _check_video_batch_contract(batch) -> None:
    sample_ids = list(batch.sample_ids)
    clip_ids = list(batch.clip_ids)
    media_uris = list(batch.media_uris)
    clip_starts = list(batch.clip_starts)
    offsets = [int(x) for x in batch.offsets]
    payload = bytes(batch.payload)

    if not sample_ids:
        raise RuntimeError("video batch missing sample_ids")
    if not clip_ids:
        raise RuntimeError("video batch missing clip_ids")
    if len(sample_ids) != len(clip_ids):
        raise RuntimeError("video batch sample_ids/clip_ids length mismatch")
    if len(media_uris) != len(clip_ids):
        raise RuntimeError("video batch media_uris/clip_ids length mismatch")
    if len(clip_starts) != len(clip_ids):
        raise RuntimeError("video batch clip_starts/clip_ids length mismatch")
    if len(offsets) != len(clip_ids) + 1:
        raise RuntimeError("video batch offsets length mismatch")
    if offsets[0] != 0:
        raise RuntimeError("video batch offsets must start at 0")
    if offsets[-1] != len(payload):
        raise RuntimeError("video batch offsets tail must equal payload length")
    for i in range(len(offsets) - 1):
        if offsets[i] > offsets[i + 1]:
            raise RuntimeError("video batch offsets must be monotonic")


def _collect_sequence(loader, max_batches: int) -> list[str]:
    seq = []
    for _idx, batch in zip(range(max_batches), loader):
        _check_video_batch_contract(batch)
        seq.extend(str(x) for x in batch.clip_ids)
    return seq


def _write_stage1_failure_manifest(path: Path) -> None:
    rows = [
        "0\ts3://bucket/video_ok.mp4\t\t\tmx8:video;frames=17;stream_id=0;codec=h264",
        "1\ts3://bucket/video_corrupt.mp4\t\t\tmx8:video;frames=25;stream_id=0;corrupt=true",
        "2\ts3://bucket/video_short.mp4\t\t\tmx8:video;frames=2;stream_id=0;codec=h264",
        "3\ts3://bucket/video_codec.mp4\t\t\tmx8:video;frames=25;stream_id=0;codec=unsupported",
        "4\ts3://bucket/video_missing_stream.mp4\t\t\tmx8:video;frames=25;codec=h264",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _check_stage1_failure_taxonomy(store_root: Path, tmp_root: Path) -> dict:
    manifest = tmp_root / "video_stage2b_failures.tsv"
    _write_stage1_failure_manifest(manifest)
    out = mx8._internal.video_index_build(
        "s3://bucket/video_stage2b/@refresh",
        manifest_store_root=str(store_root),
        dev_manifest_path=str(manifest),
        recursive=True,
        clip_len=4,
        stride=2,
        fps_policy="fixed_fps:8",
        seed=11,
        epoch=2,
        max_clips_in_memory=10_000,
    )
    failures = dict(out.get("failure_counts", {}))
    expected = {
        "corrupt_media": 1,
        "short_media": 1,
        "unsupported_codec": 1,
        "missing_stream": 1,
    }
    for key, want in expected.items():
        got = int(failures.get(key, -1))
        if got != want:
            raise RuntimeError(
                f"stage1 failure taxonomy mismatch for {key}: got={got} want={want}"
            )
    return failures


def _check_runtime_io_failure(
    store_root: Path, data_root: Path, clip_len: int, stride: int, fps: int, seed: int, epoch: int
) -> str:
    constraints = mx8.Constraints(
        max_inflight_bytes=8 * 1024 * 1024,
        max_process_rss_bytes=None,
    )
    doomed = _make_failure_probe_video(data_root)
    loader = mx8.video(
        str(data_root),
        manifest_store_root=str(store_root),
        recursive=True,
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        batch_size_samples=1,
        seed=seed,
        epoch=epoch,
        constraints=constraints,
    )
    doomed.unlink()

    try:
        for _ in range(64):
            next(loader)
    except RuntimeError as e:
        msg = str(e)
        if "video decode read failed" in msg:
            return "io_read_failed"
        raise RuntimeError(f"unexpected runtime decode failure: {msg}")
    raise RuntimeError("expected runtime IO decode failure but loader did not fail")


def main() -> None:
    tmp_root = Path(os.environ["MX8_VIDEO_STAGE2B_TMP_ROOT"])
    store_root = tmp_root / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    ds_root = tmp_root / "videos"
    _make_local_videos(ds_root)

    os.environ["MX8_VIDEO_STAGE1_INDEX"] = "1"
    os.environ["MX8_VIDEO_STAGE1_DISABLE_FFPROBE"] = "1"
    os.environ["MX8_VIDEO_STAGE2_BYTES_PER_CLIP"] = os.environ.get(
        "MX8_VIDEO_STAGE2_BYTES_PER_CLIP", "2048"
    )

    clip_len = int(os.environ.get("MX8_VIDEO_STAGE2B_CLIP_LEN", "4"))
    stride = int(os.environ.get("MX8_VIDEO_STAGE2B_STRIDE", "2"))
    fps = int(os.environ.get("MX8_VIDEO_STAGE2B_FPS", "8"))
    batch_size = int(os.environ.get("MX8_VIDEO_STAGE2B_BATCH_SIZE", "4"))
    max_batches = int(os.environ.get("MX8_VIDEO_STAGE2B_MAX_BATCHES", "12"))
    seed = int(os.environ.get("MX8_VIDEO_STAGE2B_SEED", "31"))
    epoch = int(os.environ.get("MX8_VIDEO_STAGE2B_EPOCH", "7"))

    constraints = mx8.Constraints(
        max_inflight_bytes=128 * 1024,
        max_process_rss_bytes=None,
    )

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
        raise RuntimeError("video stage2b replay mismatch for same seed/epoch")

    if int(stats1.get("video_delivered_batches_total", 0)) <= 0:
        raise RuntimeError(f"unexpected stats: {stats1}")
    if int(stats1.get("video_delivered_samples_total", 0)) <= 0:
        raise RuntimeError(f"unexpected stats: {stats1}")
    if int(stats1.get("video_delivered_bytes_total", 0)) <= 0:
        raise RuntimeError(f"unexpected stats: {stats1}")

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
            constraints=mx8.Constraints(
                max_inflight_bytes=2 * 1024,
                max_process_rss_bytes=None,
            ),
        )
        raise RuntimeError("expected max_inflight_bytes guard to reject config")
    except ValueError as e:
        if "exceeds max_inflight_bytes" not in str(e):
            raise RuntimeError(f"unexpected config error: {e}")

    runtime_failure_class = _check_runtime_io_failure(
        store_root=store_root,
        data_root=tmp_root / "runtime_failure_probe",
        clip_len=clip_len,
        stride=stride,
        fps=fps,
        seed=seed,
        epoch=epoch,
    )
    taxonomy = _check_stage1_failure_taxonomy(store_root=store_root, tmp_root=tmp_root)

    print("video_stage2b_gate_summary")
    print("  batches:", int(stats1["video_delivered_batches_total"]))
    print("  samples:", int(stats1["video_delivered_samples_total"]))
    print("  bytes:", int(stats1["video_delivered_bytes_total"]))
    print("  clips_total:", int(stats1["clips_total"]))
    print("  replay_determinism: ok")
    print("  memory_bound_init_gate: ok")
    print("  runtime_failure_class:", runtime_failure_class)
    print("  stage1_failure_taxonomy:", taxonomy)


if __name__ == "__main__":
    main()
