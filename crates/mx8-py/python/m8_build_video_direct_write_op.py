import argparse
import os
import tempfile
from pathlib import Path


def _find_library(build_root: Path, stem: str) -> Path:
    suffixes = (".so", ".dylib", ".pyd")
    candidates = []
    for path in build_root.rglob("*"):
        if not path.is_file():
            continue
        if stem not in path.name:
            continue
        if path.suffix.lower() not in suffixes:
            continue
        candidates.append(path)
    if not candidates:
        raise RuntimeError(f"no built library found under {build_root} for stem={stem}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    import sys

    import torch
    from torch.utils.cpp_extension import load

    parser = argparse.ArgumentParser(
        description="Build mx8_video direct-write Torch custom op library"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(tempfile.mkdtemp(prefix="mx8-video-direct-write-op-")),
        help="Build output directory (default: temp dir)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose build output",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    py_bin_dir = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = py_bin_dir + os.pathsep + os.environ.get("PATH", "")
    cpp_src = Path(__file__).with_name("mx8_video_direct_write_op.cpp").resolve()
    if not cpp_src.exists():
        raise RuntimeError(f"missing source file: {cpp_src}")

    built = load(
        name="mx8_video_direct_write_op",
        sources=[str(cpp_src)],
        build_directory=str(out_dir),
        extra_cflags=["-O3", "-std=c++17"],
        is_python_module=False,
        verbose=args.verbose,
    )

    if isinstance(built, str):
        lib_path = Path(built).resolve()
    else:
        lib_path = _find_library(out_dir, "mx8_video_direct_write_op")
    if not lib_path.exists():
        raise RuntimeError(f"built library path does not exist: {lib_path}")

    torch.ops.load_library(str(lib_path))
    ns = getattr(torch.ops, "mx8_video", None)
    op = getattr(ns, "direct_write_u8", None) if ns is not None else None
    decode_op = getattr(ns, "decode_file_nvdec_into_u8", None) if ns is not None else None
    if op is None:
        raise RuntimeError("torch.ops.mx8_video.direct_write_u8 was not registered")
    if decode_op is None:
        raise RuntimeError("torch.ops.mx8_video.decode_file_nvdec_into_u8 was not registered")

    print(str(lib_path))


if __name__ == "__main__":
    main()
