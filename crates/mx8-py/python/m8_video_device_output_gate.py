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


def _make_mp4(path: Path, color: str, *, fps: int = 8, seconds: int = 2) -> None:
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
        f"color=c={color}:s=128x128:r={fps}:d={seconds}",
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
    _make_mp4(root / "clip_a.mp4", "red")
    _make_mp4(root / "clip_b.mp4", "green")
    _make_mp4(root / "nested" / "clip_c.mp4", "blue")


def main() -> None:
    import torch

    tmp_root = Path(os.environ["MX8_VIDEO_DEVICE_OUTPUT_TMP_ROOT"])
    store_root = tmp_root / "store"
    data_root = tmp_root / "videos"
    store_root.mkdir(parents=True, exist_ok=True)
    _make_local_videos(data_root)

    os.environ["MX8_VIDEO_STAGE1_INDEX"] = "1"
    os.environ["MX8_VIDEO_STAGE2_BYTES_PER_CLIP"] = os.environ.get(
        "MX8_VIDEO_STAGE2_BYTES_PER_CLIP", "4096"
    )
    os.environ["MX8_VIDEO_EXPERIMENTAL_DEVICE_OUTPUT"] = "1"
    os.environ["MX8_VIDEO_EXPERIMENTAL_DEVICE_DIRECT_WRITE"] = "1"
    op_library = os.environ.get("MX8_VIDEO_DIRECT_WRITE_OP_LIBRARY", "").strip()
    if op_library:
        torch.ops.load_library(op_library)
        if not hasattr(torch.ops, "mx8_video") or not hasattr(
            torch.ops.mx8_video, "direct_write_u8"
        ):
            raise RuntimeError(
                "expected torch.ops.mx8_video.direct_write_u8 to be registered after load_library"
            )
        if torch.cuda.is_available():
            src = torch.arange(24, dtype=torch.uint8, device="cpu").view(1, 2, 2, 2, 3)
            dst = torch.empty_like(src, device="cuda")
            stream_id = int(torch.cuda.current_stream().cuda_stream)
            torch.ops.mx8_video.direct_write_u8(dst, src, stream_id)
            if not torch.equal(dst.cpu(), src):
                raise RuntimeError("direct_write_u8 sanity check failed: dst bytes differ from src")

    loader = mx8.video(
        str(data_root),
        manifest_store=str(store_root),
        recursive=True,
        clip_len=4,
        stride=2,
        fps=8,
        batch_size_samples=2,
        max_ram_gb=4,
        profile="balanced",
        autotune=True,
    )

    batch = next(iter(loader))
    payload_u8, offsets_i64, sample_ids_i64 = batch.to_torch()
    stats = dict(loader.stats())

    if payload_u8.dtype != torch.uint8:
        raise RuntimeError(f"expected payload dtype uint8, got {payload_u8.dtype}")
    if payload_u8.dim() != 5:
        raise RuntimeError(f"expected 5D payload tensor, got shape {tuple(payload_u8.shape)}")
    if int(payload_u8.shape[0]) != int(sample_ids_i64.numel()):
        raise RuntimeError(
            "expected payload batch dimension to match sample_ids length "
            f"(shape0={payload_u8.shape[0]}, sample_ids={sample_ids_i64.numel()})"
        )
    if int(offsets_i64[-1].item()) != int(payload_u8.numel()):
        raise RuntimeError(
            "expected offsets tail to match payload numel "
            f"(offsets_tail={offsets_i64[-1].item()}, payload_numel={payload_u8.numel()})"
        )
    if int(payload_u8.data_ptr()) == 0:
        raise RuntimeError("expected non-zero payload data_ptr")

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
            "expected contiguous THWC stride contract "
            f"(actual={actual_strides}, expected={expected_strides})"
        )
    if not bool(payload_u8.is_contiguous()):
        raise RuntimeError("expected contiguous payload tensor")

    expected_active = bool(torch.cuda.is_available())
    requested = bool(stats.get("video_experimental_device_output_requested", False))
    active = bool(stats.get("video_experimental_device_output_active", False))
    fallback_total = int(stats.get("video_experimental_device_output_fallback_total", 0))
    direct_requested = bool(
        stats.get("video_experimental_device_direct_write_requested", False)
    )
    direct_active = bool(stats.get("video_experimental_device_direct_write_active", False))
    direct_fallback_total = int(
        stats.get("video_experimental_device_direct_write_fallback_total", 0)
    )
    direct_batches_total = int(
        stats.get("video_experimental_device_direct_write_batches_total", 0)
    )
    if not requested:
        raise RuntimeError("expected video_experimental_device_output_requested=true")
    if active != expected_active:
        raise RuntimeError(
            "device-output activation mismatch "
            f"(expected_active={expected_active}, stats_active={active})"
        )
    if expected_active:
        if payload_u8.device.type != "cuda":
            raise RuntimeError(f"expected CUDA payload tensor, got {payload_u8.device}")
        if fallback_total != 0:
            raise RuntimeError(f"expected zero fallback_total when CUDA active, got {fallback_total}")
        if not direct_requested:
            raise RuntimeError("expected direct-write requested=true")
        if not direct_active:
            raise RuntimeError("expected direct-write active=true when CUDA active")
        if direct_fallback_total != 0:
            raise RuntimeError(
                f"expected direct-write fallback_total=0, got {direct_fallback_total}"
            )
        if direct_batches_total <= 0:
            raise RuntimeError("expected direct-write batches_total > 0 when active")
    else:
        if payload_u8.device.type != "cpu":
            raise RuntimeError(f"expected CPU payload tensor fallback, got {payload_u8.device}")
        if fallback_total <= 0:
            raise RuntimeError("expected fallback_total > 0 when CUDA unavailable")
        if not direct_requested:
            raise RuntimeError("expected direct-write requested=true")
        if direct_active:
            raise RuntimeError("expected direct-write active=false when CUDA unavailable")
        if direct_fallback_total <= 0:
            raise RuntimeError("expected direct-write fallback_total > 0 when inactive")
        if direct_batches_total != 0:
            raise RuntimeError("expected direct-write batches_total == 0 when inactive")

    print("video_device_output_gate_summary")
    print(f"  torch_cuda_available: {expected_active}")
    print(f"  payload_device: {payload_u8.device}")
    print(f"  requested: {requested}")
    print(f"  active: {active}")
    print(f"  fallback_total: {fallback_total}")
    print(f"  direct_requested: {direct_requested}")
    print(f"  direct_active: {direct_active}")
    print(f"  direct_fallback_total: {direct_fallback_total}")
    print(f"  direct_batches_total: {direct_batches_total}")
    print(f"  payload_shape: {tuple(payload_u8.shape)}")
    print(f"  sample_ids: {int(sample_ids_i64.numel())}")


if __name__ == "__main__":
    main()
