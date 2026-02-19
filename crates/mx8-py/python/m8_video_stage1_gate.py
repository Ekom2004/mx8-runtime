import os
from pathlib import Path

import mx8


def _write_dev_manifest(path: Path) -> None:
    rows = [
        "0\ts3://bucket/video_ok.mp4\t\t\tmx8:video;frames=17;stream_id=0;codec=h264",
        "1\ts3://bucket/video_corrupt.mp4\t\t\tmx8:video;frames=25;stream_id=0;corrupt=true",
        "2\ts3://bucket/video_short.mp4\t\t\tmx8:video;frames=2;stream_id=0;codec=h264",
        "3\ts3://bucket/video_codec.mp4\t\t\tmx8:video;frames=25;stream_id=0;codec=unsupported",
        "4\ts3://bucket/video_missing_stream.mp4\t\t\tmx8:video;frames=25;codec=h264",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    store_root = Path(os.environ["MX8_MANIFEST_STORE_ROOT"])
    dev_manifest = Path(os.environ["MX8_DEV_MANIFEST_PATH"])
    dataset_link = os.environ.get("MX8_DATASET_LINK", "s3://bucket/videos/@refresh")
    store_root.mkdir(parents=True, exist_ok=True)
    _write_dev_manifest(dev_manifest)

    out = mx8._internal.video_index_build(
        dataset_link,
        manifest_store_root=str(store_root),
        dev_manifest_path=str(dev_manifest),
        recursive=True,
        clip_len=4,
        stride=2,
        fps_policy="fixed_fps:8",
        seed=19,
        epoch=3,
        max_clips_in_memory=10_000,
    )

    if int(out.get("video_schema_version", 0)) != 1:
        raise RuntimeError(f"unexpected video schema version: {out}")
    clip_count = int(out.get("clip_count", 0))
    if clip_count <= 0:
        raise RuntimeError(f"expected clip_count > 0, got {clip_count}")
    clip_index_hash = str(out.get("clip_index_hash", ""))
    if not clip_index_hash:
        raise RuntimeError("clip_index_hash is empty")

    failures = out.get("failure_counts", {})
    expected = {
        "corrupt_media": 1,
        "short_media": 1,
        "unsupported_codec": 1,
        "missing_stream": 1,
    }
    for key, want in expected.items():
        got = int(failures.get(key, -1))
        if got != want:
            raise RuntimeError(f"failure counter mismatch for {key}: got={got} want={want}")

    replay = mx8._internal.video_index_replay_check(
        dataset_link,
        manifest_store_root=str(store_root),
        dev_manifest_path=str(dev_manifest),
        recursive=True,
        clip_len=4,
        stride=2,
        fps_policy="fixed_fps:8",
        seed=19,
        epoch=3,
        max_clips_in_memory=10_000,
    )
    if not bool(replay.get("deterministic", False)):
        raise RuntimeError(f"replay check failed: {replay}")

    try:
        mx8._internal.video_index_build(
            dataset_link,
            manifest_store_root=str(store_root),
            dev_manifest_path=str(dev_manifest),
            recursive=True,
            clip_len=4,
            stride=2,
            fps_policy="fixed_fps:8",
            seed=19,
            epoch=3,
            max_clips_in_memory=2,
        )
        raise RuntimeError("expected memory-bound failure but build succeeded")
    except RuntimeError as e:
        if "memory bound exceeded" not in str(e):
            raise RuntimeError(f"expected memory-bound error, got: {e}")

    print("video_stage1_gate_summary")
    print("  clip_count:", clip_count)
    print("  clip_index_hash:", clip_index_hash)
    print("  tail_clips_dropped_total:", int(out.get("tail_clips_dropped_total", 0)))
    print("  failure_counts:", failures)
    print("  replay_determinism: ok")
    print("  memory_bound_gate: ok")
    print("  schema_version_gate: ok")

    # Local-prefix metadata extraction integration check (Stage 1 real path):
    # enable video hint injection from local file metadata/probe and verify
    # clip index can be built without dev manifest hints.
    local_dir = store_root.parent / "local_video_probe"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_mp4 = local_dir / "clip_a.mp4"
    local_mp4.write_bytes(b"\x00" * 400_000)
    local_out = mx8._internal.video_index_build(
        str(local_dir),
        manifest_store_root=str(store_root),
        recursive=True,
        clip_len=4,
        stride=2,
        fps_policy="fixed_fps:8",
        seed=23,
        epoch=5,
        max_clips_in_memory=10_000,
    )
    local_clip_count = int(local_out.get("clip_count", 0))
    if local_clip_count <= 0:
        raise RuntimeError(f"expected local probe clip_count > 0, got {local_out}")
    print("  local_metadata_probe_gate: ok")


if __name__ == "__main__":
    main()
