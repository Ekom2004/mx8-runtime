# Video GA Checklist

This checklist is the go/no-go contract for moving MX8 video from preview to GA.

## GA bar (must all pass)

- **Correctness:** clip decode contract and failure taxonomy are stable.
- **Determinism:** replay on fixed seed/dataset is stable.
- **Boundedness:** decode/runtime stays within configured caps.
- **Reliability:** decode-path failures are classified and surfaced in stats/logs.
- **Performance floor:** decode throughput stays above agreed baseline.
- **Range planning:** Stage2D planner contract remains deterministic and schema-pinned.
- **Backend safety:** backend selector works and `ffi` fallback to `cli` is safe.

## One-command gate

Run:

```bash
./scripts/video_ga_gate.sh --full
```

This executes the required stage gates and fails fast on first regression.

## Required gate set

- `./scripts/video_stage1_gate.sh`
- `./scripts/video_stage2a_gate.sh`
- `./scripts/video_stage2b_clean_env_gate.sh`
- `./scripts/video_stage2c_perf_gate.sh`
- `./scripts/video_stage2d_range_gate.sh`
- `./scripts/video_stage3a_backend_gate.sh`

## Fast precheck (developer loop)

Run:

```bash
./scripts/video_ga_gate.sh --quick
```

This runs a smaller subset for iteration before full GA validation.

## Release decision

Video is GA-ready when:

- `video_ga_gate.sh --full` passes on a clean environment run.
- no open P0/P1 video bugs are outstanding.
- README/API docs no longer label video as preview.
