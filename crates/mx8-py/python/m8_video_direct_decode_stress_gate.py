import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import mx8


def _ffmpeg_bin() -> str:
    ffmpeg_bin = os.environ.get("MX8_FFMPEG_BIN", "ffmpeg")
    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(f"ffmpeg not found on PATH (MX8_FFMPEG_BIN={ffmpeg_bin})")
    return ffmpeg_bin


def _make_mp4(path: Path, filter_expr: str) -> None:
    ffmpeg_bin = _ffmpeg_bin()
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        filter_expr,
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed creating {path}: {proc.stderr.strip() or proc.stdout.strip()}"
        )


def _make_local_videos(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _make_mp4(root / "clip_a.mp4", "testsrc=size=128x128:rate=8:duration=2")
    _make_mp4(root / "clip_b.mp4", "testsrc2=size=128x128:rate=8:duration=2")
    _make_mp4(root / "nested" / "clip_c.mp4", "smptebars=size=128x128:rate=8:duration=2")
    _make_mp4(root / "nested" / "clip_d.mp4", "pal100bars=size=128x128:rate=8:duration=2")


def _collect_batches(data_root: Path, store_root: Path, *, direct_decode: bool):
    import torch

    os.environ["MX8_VIDEO_STAGE1_INDEX"] = "1"
    os.environ["MX8_VIDEO_STAGE2_BYTES_PER_CLIP"] = os.environ.get(
        "MX8_VIDEO_STAGE2_BYTES_PER_CLIP", "4096"
    )
    os.environ["MX8_VIDEO_DECODE_BACKEND"] = "nvdec"
    os.environ["MX8_VIDEO_EXPERIMENTAL_DEVICE_OUTPUT"] = "1"
    os.environ["MX8_VIDEO_EXPERIMENTAL_DEVICE_DIRECT_WRITE"] = "1"
    os.environ["MX8_VIDEO_EXPERIMENTAL_DIRECT_DECODE_TO_DESTINATION"] = (
        "1" if direct_decode else "0"
    )

    loader = mx8.video(
        str(data_root),
        store=str(store_root),
        recursive=True,
        clip=4,
        stride=2,
        fps=8,
        batch=2,
        ram_gb=4,
        profile="balanced",
        tune=True,
    )
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    digests = []
    for batch_idx, batch in enumerate(loader):
        stream = streams[batch_idx % len(streams)]
        with torch.cuda.stream(stream):
            before = int(torch.cuda.current_stream().cuda_stream)
            payload_u8, offsets_i64, sample_ids_i64 = batch.to_torch()
            after = int(torch.cuda.current_stream().cuda_stream)
            if before != after:
                raise RuntimeError(
                    f"stream changed during to_torch() (before={before}, after={after})"
                )
            if payload_u8.device.type != "cuda":
                raise RuntimeError(f"expected CUDA payload tensor, got {payload_u8.device}")
            if not bool(payload_u8.is_contiguous()):
                raise RuntimeError("expected contiguous payload tensor")
            shape = [int(v) for v in payload_u8.shape]
            expected_strides = (
                shape[1] * shape[2] * shape[3] * shape[4],
                shape[2] * shape[3] * shape[4],
                shape[3] * shape[4],
                shape[4],
                1,
            )
            actual_strides = tuple(int(v) for v in payload_u8.stride())
            if actual_strides != expected_strides:
                raise RuntimeError(
                    "stride contract mismatch "
                    f"(actual={actual_strides}, expected={expected_strides})"
                )
            torch.cuda.current_stream().synchronize()
            payload_cpu = payload_u8.cpu()
        sample_ids = tuple(int(v) for v in sample_ids_i64.tolist())
        offsets = tuple(int(v) for v in offsets_i64.tolist())
        flat_u8 = bytes(int(v) for v in payload_cpu.contiguous().view(-1).tolist())
        payload_digest = hashlib.sha256(flat_u8).hexdigest()
        digests.append((sample_ids, offsets, payload_digest))
    return digests, dict(loader.stats())


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("video_direct_decode_stress_gate requires CUDA")
    op_library = os.environ.get("MX8_VIDEO_DIRECT_WRITE_OP_LIBRARY", "").strip()
    if not op_library:
        raise RuntimeError("set MX8_VIDEO_DIRECT_WRITE_OP_LIBRARY to run this gate")
    torch.ops.load_library(op_library)
    if (
        not hasattr(torch.ops, "mx8_video")
        or not hasattr(torch.ops.mx8_video, "direct_write_u8")
        or not hasattr(torch.ops.mx8_video, "decode_file_nvdec_into_u8")
    ):
        raise RuntimeError("expected mx8_video custom ops to be registered")

    tmp_root = Path(os.environ["MX8_VIDEO_DEVICE_OUTPUT_TMP_ROOT"])
    store_root = tmp_root / "store"
    data_root = tmp_root / "videos"
    store_root.mkdir(parents=True, exist_ok=True)
    _make_local_videos(data_root)

    baseline_digests, baseline_stats = _collect_batches(
        data_root, store_root / "baseline", direct_decode=False
    )
    direct_digests, direct_stats = _collect_batches(
        data_root, store_root / "direct", direct_decode=True
    )

    if baseline_digests != direct_digests:
        raise RuntimeError("direct-decode payload parity mismatch vs baseline path")
    if not bool(direct_stats.get("video_experimental_direct_decode_to_destination_requested", False)):
        raise RuntimeError("expected direct-decode requested=true")
    if not bool(direct_stats.get("video_experimental_direct_decode_to_destination_active", False)):
        raise RuntimeError("expected direct-decode active=true")
    if int(direct_stats.get("video_experimental_direct_decode_to_destination_fallback_total", 0)) != 0:
        raise RuntimeError("expected direct-decode fallback_total=0 in stress gate")
    if int(direct_stats.get("video_experimental_direct_decode_to_destination_batches_total", 0)) <= 0:
        raise RuntimeError("expected direct-decode batches_total > 0")

    print("video_direct_decode_stress_gate_summary")
    print(f"  batches: {len(direct_digests)}")
    print(f"  parity: ok")
    print(f"  direct_decode_active: {direct_stats.get('video_experimental_direct_decode_to_destination_active')}")
    print(f"  direct_decode_batches_total: {direct_stats.get('video_experimental_direct_decode_to_destination_batches_total')}")
    print(f"  decode_backend: {direct_stats.get('video_decode_backend')}")


if __name__ == "__main__":
    main()
