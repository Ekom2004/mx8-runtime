#include <cstdint>
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
}  // namespace

TORCH_LIBRARY(mx8_video, m) {
  m.def("direct_write_u8(Tensor dst, Tensor src, int stream_id) -> ()");
}

TORCH_LIBRARY_IMPL(mx8_video, CompositeExplicitAutograd, m) {
  m.impl("direct_write_u8", direct_write_u8);
}
