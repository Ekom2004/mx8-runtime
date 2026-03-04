#include <torch/extension.h>

namespace {
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
  (void)stream_id;
  dst.copy_(src, /*non_blocking=*/true);
}
}  // namespace

TORCH_LIBRARY(mx8_video, m) {
  m.def("direct_write_u8(Tensor dst, Tensor src, int stream_id) -> ()");
}

TORCH_LIBRARY_IMPL(mx8_video, CompositeExplicitAutograd, m) {
  m.impl("direct_write_u8", direct_write_u8);
}

