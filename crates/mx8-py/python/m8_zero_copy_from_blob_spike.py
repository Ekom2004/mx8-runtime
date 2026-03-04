import gc
import shutil
import tempfile
import time
from pathlib import Path


CPP_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace py = pybind11;

struct AllocationState {
  bool consumer_done = false;
  bool deleter_called = false;
  bool deleter_before_consumer_done = false;
  int64_t alloc_ns = 0;
  int64_t done_ns = 0;
  int64_t deleter_ns = 0;
  int64_t deleter_calls = 0;
};

static std::mutex g_mu;
static std::unordered_map<int64_t, AllocationState> g_states;
static int64_t g_next_id = 1;

static int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::tuple<torch::Tensor, int64_t> make_blob_tensor(int64_t numel, int64_t device_index) {
  if (numel <= 0) {
    throw std::runtime_error("numel must be > 0");
  }
  const size_t bytes = static_cast<size_t>(numel) * sizeof(float);
  void* ptr = c10::cuda::CUDACachingAllocator::raw_alloc(bytes);
  int64_t alloc_id = 0;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    alloc_id = g_next_id++;
    AllocationState state;
    state.alloc_ns = now_ns();
    g_states[alloc_id] = state;
  }

  auto deleter = [alloc_id](void* p) {
    {
      std::lock_guard<std::mutex> lock(g_mu);
      auto it = g_states.find(alloc_id);
      if (it != g_states.end()) {
        AllocationState& state = it->second;
        state.deleter_calls += 1;
        state.deleter_called = true;
        state.deleter_ns = now_ns();
        if (!state.consumer_done) {
          state.deleter_before_consumer_done = true;
        }
      }
    }
    c10::cuda::CUDACachingAllocator::raw_delete(p);
  };

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_index));
  auto tensor = torch::from_blob(ptr, {numel}, deleter, options);
  return std::make_tuple(tensor, alloc_id);
}

void mark_consumer_done(int64_t alloc_id) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_states.find(alloc_id);
  if (it == g_states.end()) {
    throw std::runtime_error("unknown alloc_id");
  }
  it->second.consumer_done = true;
  it->second.done_ns = now_ns();
}

py::dict get_state(int64_t alloc_id) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto it = g_states.find(alloc_id);
  if (it == g_states.end()) {
    throw std::runtime_error("unknown alloc_id");
  }
  const AllocationState& state = it->second;
  py::dict out;
  out["consumer_done"] = state.consumer_done;
  out["deleter_called"] = state.deleter_called;
  out["deleter_before_consumer_done"] = state.deleter_before_consumer_done;
  out["alloc_ns"] = state.alloc_ns;
  out["done_ns"] = state.done_ns;
  out["deleter_ns"] = state.deleter_ns;
  out["deleter_calls"] = state.deleter_calls;
  return out;
}

void erase_state(int64_t alloc_id) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_states.erase(alloc_id);
}

void reset_states() {
  std::lock_guard<std::mutex> lock(g_mu);
  g_states.clear();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_blob_tensor", &make_blob_tensor, "Allocate CUDA blob tensor with custom deleter");
  m.def("mark_consumer_done", &mark_consumer_done, "Mark consumer completion for an allocation");
  m.def("get_state", &get_state, "Inspect allocation/deleter state");
  m.def("erase_state", &erase_state, "Erase allocation state");
  m.def("reset_states", &reset_states, "Reset all allocation states");
}
"""


def _build_extension(build_root: Path):
    import torch
    from torch.utils.cpp_extension import load_inline

    build_root.mkdir(parents=True, exist_ok=True)
    name = f"mx8_zero_copy_spike_ext_{int(time.time())}"
    return load_inline(
        name=name,
        cpp_sources=[CPP_SRC],
        functions=None,
        with_cuda=True,
        extra_cflags=["-O2", "-std=c++17"],
        build_directory=str(build_root),
        verbose=False,
    )


def _wait_for_deleter(ext, alloc_id: int, timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    last_state = {}
    while time.time() < deadline:
        gc.collect()
        state = dict(ext.get_state(alloc_id))
        last_state = state
        if bool(state.get("deleter_called", False)):
            return state
        time.sleep(0.01)
    return last_state


def _run_case(ext, *, explicit_wait: bool, iterations: int, numel: int) -> dict:
    import torch

    failures = 0
    not_called = 0
    early_called = 0
    total = 0
    for _ in range(iterations):
        total += 1
        producer = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        tensor, alloc_id = ext.make_blob_tensor(numel, torch.cuda.current_device())

        with torch.cuda.stream(producer):
            tensor.fill_(1.0)
            for _inner in range(5):
                tensor.mul_(1.0001).sin_()

        done_event = torch.cuda.Event(enable_timing=False, blocking=False)
        with torch.cuda.stream(consumer):
            if explicit_wait:
                consumer.wait_stream(producer)
            # Force consumer stream to read from the same from_blob storage.
            m = tensor.view(4096, -1)
            z = (m * 1.0001).sum()
            done_event.record()

        # Drop python refs before work has necessarily completed.
        del tensor
        del m
        del z
        gc.collect()

        done_event.synchronize()
        ext.mark_consumer_done(alloc_id)
        state = _wait_for_deleter(ext, alloc_id, timeout_s=15.0)
        if not bool(state.get("deleter_called", False)):
            failures += 1
            not_called += 1
        elif bool(state.get("deleter_before_consumer_done", False)):
            failures += 1
            early_called += 1
        ext.erase_state(alloc_id)

    return {
        "total": total,
        "failures": failures,
        "not_called": not_called,
        "early_called": early_called,
    }


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("zero-copy spike requires CUDA (torch.cuda.is_available() is false)")
    if torch.version.cuda is None:
        raise RuntimeError("zero-copy spike requires a CUDA-enabled PyTorch build")

    tmp_root = Path(tempfile.mkdtemp(prefix="mx8-zero-copy-spike-"))
    try:
        ext = _build_extension(tmp_root / "build")
        ext.reset_states()

        iterations = 32
        numel = 1024 * 1024 * 16  # 64 MiB float32 buffer

        explicit = _run_case(ext, explicit_wait=True, iterations=iterations, numel=numel)
        implicit = _run_case(ext, explicit_wait=False, iterations=iterations, numel=numel)

        print("zero_copy_from_blob_spike_summary")
        print(f"  torch_cuda_version: {torch.version.cuda}")
        print(f"  torch_device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"  explicit_wait_total: {explicit['total']}")
        print(f"  explicit_wait_failures: {explicit['failures']}")
        print(f"  explicit_wait_not_called: {explicit['not_called']}")
        print(f"  explicit_wait_early_called: {explicit['early_called']}")
        print(f"  implicit_wait_total: {implicit['total']}")
        print(f"  implicit_wait_failures: {implicit['failures']}")
        print(f"  implicit_wait_not_called: {implicit['not_called']}")
        print(f"  implicit_wait_early_called: {implicit['early_called']}")

        if explicit["failures"] > 0:
            raise RuntimeError(
                "zero-copy spike failed: deleter safety violation in explicit-wait case"
            )
        if implicit["failures"] > 0:
            raise RuntimeError("zero-copy spike failed: deleter safety violation in implicit case")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
