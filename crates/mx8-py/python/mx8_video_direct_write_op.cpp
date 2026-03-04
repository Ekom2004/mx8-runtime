#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <limits>

#include <torch/extension.h>

#if defined(__has_include)
#if __has_include(<cuda_runtime_api.h>)
#define MX8_VIDEO_DIRECT_WRITE_HAS_CUDA_RUNTIME 1
#include <cuda_runtime_api.h>
#else
#define MX8_VIDEO_DIRECT_WRITE_HAS_CUDA_RUNTIME 0
#endif
#else
#define MX8_VIDEO_DIRECT_WRITE_HAS_CUDA_RUNTIME 0
#endif

namespace {
std::string shell_escape_single_quotes(const std::string& input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (char ch : input) {
    if (ch == '\'') {
      out += "'\"'\"'";
    } else {
      out.push_back(ch);
    }
  }
  return out;
}

size_t checked_num_bytes(const torch::Tensor& src) {
  const auto numel = src.numel();
  TORCH_CHECK(numel >= 0, "mx8_video::direct_write_u8: src numel must be non-negative");
  const auto element_size = src.element_size();
  const auto numel_u64 = static_cast<uint64_t>(numel);
  const auto element_u64 = static_cast<uint64_t>(element_size);
  TORCH_CHECK(
      element_u64 == 1,
      "mx8_video::direct_write_u8: expected uint8 element size=1, got ",
      element_u64);
  TORCH_CHECK(
      numel_u64 <= (std::numeric_limits<size_t>::max() / element_u64),
      "mx8_video::direct_write_u8: byte-size overflow for src tensor");
  return static_cast<size_t>(numel_u64 * element_u64);
}

void direct_write_u8(torch::Tensor dst, torch::Tensor src, int64_t stream_id) {
  TORCH_CHECK(dst.is_cuda(), "mx8_video::direct_write_u8: dst must be a CUDA tensor");
  TORCH_CHECK(dst.scalar_type() == torch::kUInt8,
              "mx8_video::direct_write_u8: dst dtype must be uint8");
  TORCH_CHECK(src.scalar_type() == torch::kUInt8,
              "mx8_video::direct_write_u8: src dtype must be uint8");
  TORCH_CHECK(dst.sizes() == src.sizes(),
              "mx8_video::direct_write_u8: shape mismatch");
  TORCH_CHECK(dst.is_contiguous(),
              "mx8_video::direct_write_u8: dst must be contiguous");
  TORCH_CHECK(src.is_contiguous(),
              "mx8_video::direct_write_u8: src must be contiguous");
  TORCH_CHECK(stream_id >= 0,
              "mx8_video::direct_write_u8: stream_id must be >= 0");
  const auto byte_count = checked_num_bytes(src);
  if (byte_count == 0) {
    return;
  }

#if MX8_VIDEO_DIRECT_WRITE_HAS_CUDA_RUNTIME
  auto* dst_ptr = dst.data_ptr<uint8_t>();
  auto* src_ptr = src.data_ptr<uint8_t>();
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_id));
  if (src.is_cuda()) {
    TORCH_CHECK(
        src.get_device() == dst.get_device(),
        "mx8_video::direct_write_u8: src/dst CUDA device mismatch (src=",
        src.get_device(),
        ", dst=",
        dst.get_device(),
        ")");
    const auto status =
        cudaMemcpyAsync(dst_ptr, src_ptr, byte_count, cudaMemcpyDeviceToDevice, stream);
    TORCH_CHECK(
        status == cudaSuccess,
        "mx8_video::direct_write_u8: cudaMemcpyAsync D2D failed: ",
        cudaGetErrorString(status));
    return;
  }

  const auto status =
      cudaMemcpyAsync(dst_ptr, src_ptr, byte_count, cudaMemcpyHostToDevice, stream);
  TORCH_CHECK(
      status == cudaSuccess,
      "mx8_video::direct_write_u8: cudaMemcpyAsync H2D failed: ",
      cudaGetErrorString(status));
#else
  (void)stream_id;
  // CPU-only extension build fallback: keep prior behavior so op registration
  // and non-CUDA gates continue to run.
  dst.copy_(src, /*non_blocking=*/true);
#endif
}

void decode_file_nvdec_into_u8(
    torch::Tensor dst,
    const std::string& media_path,
    double start_seconds,
    int64_t frames,
    int64_t side,
    int64_t stream_id) {
  TORCH_CHECK(
      dst.is_cuda(),
      "mx8_video::decode_file_nvdec_into_u8: dst must be a CUDA tensor");
  TORCH_CHECK(
      dst.scalar_type() == torch::kUInt8,
      "mx8_video::decode_file_nvdec_into_u8: dst dtype must be uint8");
  TORCH_CHECK(
      dst.is_contiguous(),
      "mx8_video::decode_file_nvdec_into_u8: dst must be contiguous");
  TORCH_CHECK(
      stream_id >= 0,
      "mx8_video::decode_file_nvdec_into_u8: stream_id must be >= 0");
  TORCH_CHECK(
      frames > 0 && side > 0,
      "mx8_video::decode_file_nvdec_into_u8: frames and side must be positive");

  const auto expected_bytes = checked_num_bytes(dst);
  std::ostringstream cmd;
  cmd << "ffmpeg -hide_banner -loglevel error -nostdin "
      << "-hwaccel cuda -hwaccel_output_format cuda "
      << "-ss " << start_seconds << " "
      << "-i '" << shell_escape_single_quotes(media_path) << "' "
      << "-an -sn -dn "
      << "-frames:v " << frames << " "
      << "-vf 'hwdownload,format=nv12,scale=" << side << ":" << side
      << ":flags=bilinear,format=rgb24' "
      << "-pix_fmt rgb24 -f rawvideo pipe:1";

  FILE* pipe = popen(cmd.str().c_str(), "r");
  TORCH_CHECK(
      pipe != nullptr,
      "mx8_video::decode_file_nvdec_into_u8: failed to launch ffmpeg process");

#if MX8_VIDEO_DIRECT_WRITE_HAS_CUDA_RUNTIME
  auto* dst_ptr = dst.data_ptr<uint8_t>();
  auto stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_id));
  std::vector<uint8_t> chunk(1 << 20);
  size_t copied = 0;
  while (true) {
    const size_t n = fread(chunk.data(), 1, chunk.size(), pipe);
    if (n > 0) {
      TORCH_CHECK(
          copied + n <= expected_bytes,
          "mx8_video::decode_file_nvdec_into_u8: ffmpeg produced too many bytes (",
          copied + n,
          " > ",
          expected_bytes,
          ")");
      const auto status = cudaMemcpyAsync(
          dst_ptr + copied, chunk.data(), n, cudaMemcpyHostToDevice, stream);
      TORCH_CHECK(
          status == cudaSuccess,
          "mx8_video::decode_file_nvdec_into_u8: cudaMemcpyAsync H2D failed: ",
          cudaGetErrorString(status));
      copied += n;
    }
    if (n < chunk.size()) {
      if (feof(pipe)) {
        break;
      }
      TORCH_CHECK(
          false,
          "mx8_video::decode_file_nvdec_into_u8: fread failed while reading ffmpeg stdout");
    }
  }
#else
  std::vector<uint8_t> bytes;
  bytes.resize(expected_bytes);
  const size_t copied = fread(bytes.data(), 1, expected_bytes + 1, pipe);
#endif
  const int rc = pclose(pipe);
  TORCH_CHECK(
      rc == 0,
      "mx8_video::decode_file_nvdec_into_u8: ffmpeg exited with status ",
      rc);
#if MX8_VIDEO_DIRECT_WRITE_HAS_CUDA_RUNTIME
  TORCH_CHECK(
      copied == expected_bytes,
      "mx8_video::decode_file_nvdec_into_u8: decoded byte count mismatch (got ",
      copied,
      ", expected ",
      expected_bytes,
      ")");
#else
  TORCH_CHECK(
      copied == expected_bytes,
      "mx8_video::decode_file_nvdec_into_u8: decoded byte count mismatch (got ",
      copied,
      ", expected ",
      expected_bytes,
      ")");
  auto src = torch::from_blob(
                 bytes.data(),
                 {static_cast<long long>(expected_bytes)},
                 torch::TensorOptions().dtype(torch::kUInt8))
                 .clone();
  auto dst_flat = dst.view({-1});
  dst_flat.copy_(src, /*non_blocking=*/true);
#endif
}
}  // namespace

TORCH_LIBRARY(mx8_video, m) {
  m.def("direct_write_u8(Tensor dst, Tensor src, int stream_id) -> ()");
  m.def(
      "decode_file_nvdec_into_u8(Tensor dst, str media_path, float start_seconds, int frames, int side, int stream_id) -> ()");
}

TORCH_LIBRARY_IMPL(mx8_video, CompositeExplicitAutograd, m) {
  m.impl("direct_write_u8", direct_write_u8);
  m.impl("decode_file_nvdec_into_u8", decode_file_nvdec_into_u8);
}
