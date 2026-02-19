import os

import mx8


def _source_tag(batch) -> str:
    offsets = batch.offsets
    payload = batch.payload
    if len(offsets) < 2:
        raise RuntimeError("batch offsets missing range")
    start = int(offsets[0])
    end = int(offsets[1])
    if end <= start:
        raise RuntimeError("invalid first sample offsets in mixed batch")
    first = payload[start:end][0]
    if first == ord("A"):
        return "A"
    if first == ord("B"):
        return "B"
    raise RuntimeError(f"unexpected source tag byte={first}")


def _run_once(
    *,
    store_root: str,
    link_a: str,
    link_b: str,
    manifest_a: str,
    manifest_b: str,
    seed: int,
    epoch: int,
    steps: int,
):
    loader_a = mx8.load(
        link_a,
        manifest_store_root=store_root,
        dev_manifest_path=manifest_a,
        batch_size_samples=1,
        max_inflight_bytes=8 * 1024 * 1024,
        max_queue_batches=8,
        prefetch_batches=1,
    )
    loader_b = mx8.load(
        link_b,
        manifest_store_root=store_root,
        dev_manifest_path=manifest_b,
        batch_size_samples=1,
        max_inflight_bytes=8 * 1024 * 1024,
        max_queue_batches=8,
        prefetch_batches=1,
    )

    mixed = mx8.mix(
        [loader_a, loader_b],
        weights=[1, 1],  # fairness baseline
        seed=seed,
        epoch=epoch,
        starvation_window=10_000,
    )

    seq = []
    for _, batch in zip(range(steps), mixed):
        seq.append(_source_tag(batch))
    stats = mixed.stats()
    return seq, stats


def main() -> None:
    store_root = os.environ["MX8_MANIFEST_STORE_ROOT"]
    link_a = os.environ["MX8_DATASET_LINK_A"]
    link_b = os.environ["MX8_DATASET_LINK_B"]
    manifest_a = os.environ["MX8_DEV_MANIFEST_PATH_A"]
    manifest_b = os.environ["MX8_DEV_MANIFEST_PATH_B"]

    steps = int(os.environ.get("MX8_MIX_GATE_STEPS", "160"))
    seed = int(os.environ.get("MX8_MIX_GATE_SEED", "17"))
    epoch = int(os.environ.get("MX8_MIX_GATE_EPOCH", "3"))

    seq1, stats1 = _run_once(
        store_root=store_root,
        link_a=link_a,
        link_b=link_b,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        seed=seed,
        epoch=epoch,
        steps=steps,
    )
    seq2, _stats2 = _run_once(
        store_root=store_root,
        link_a=link_a,
        link_b=link_b,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        seed=seed,
        epoch=epoch,
        steps=steps,
    )
    if seq1 != seq2:
        raise RuntimeError("mix replay mismatch for identical seed/epoch")

    seq3a, _stats3a = _run_once(
        store_root=store_root,
        link_a=link_a,
        link_b=link_b,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        seed=seed,
        epoch=epoch + 1,
        steps=steps,
    )
    seq3b, _stats3b = _run_once(
        store_root=store_root,
        link_a=link_a,
        link_b=link_b,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        seed=seed,
        epoch=epoch + 1,
        steps=steps,
    )
    if seq3a != seq3b:
        raise RuntimeError("mix replay mismatch for repeated epoch+1 run")

    realized = list(stats1.get("mix_realized_ratio", []))
    if len(realized) != 2:
        raise RuntimeError(f"unexpected realized ratio shape: {realized}")
    tol = float(os.environ.get("MX8_MIX_GATE_RATIO_TOL", "0.12"))
    if abs(realized[0] - 0.5) > tol or abs(realized[1] - 0.5) > tol:
        raise RuntimeError(
            f"mix realized ratio outside tolerance: got={realized} tol={tol}"
        )

    starvation = list(stats1.get("mix_source_starvation_total", []))
    if any(int(v) != 0 for v in starvation):
        raise RuntimeError(f"mix starvation counter non-zero: {starvation}")

    snapshot_enabled = bool(stats1.get("mix_snapshot_enabled", False))
    if not snapshot_enabled:
        raise RuntimeError("mix snapshot expected enabled but is false")
    snapshot_period = int(stats1.get("mix_snapshot_period_ticks", 0))
    if snapshot_period <= 0:
        raise RuntimeError(f"invalid mix snapshot period: {snapshot_period}")
    snapshot_emitted = int(stats1.get("mix_snapshot_emitted_total", 0))
    if snapshot_emitted <= 0:
        raise RuntimeError("mix snapshot emitted_total did not increase")
    schedule_ticks = int(stats1.get("mix_schedule_ticks", 0))
    if schedule_ticks < steps:
        raise RuntimeError(
            f"unexpected mix schedule ticks: ticks={schedule_ticks} steps={steps}"
        )

    print("mix_gate_summary")
    print("  steps:", steps)
    print("  realized_ratio:", realized)
    print("  starvation:", starvation)
    print("  snapshot_period_ticks:", snapshot_period)
    print("  snapshot_emitted_total:", snapshot_emitted)
    print("  sequence_head:", "".join(seq1[:24]))
    print("  replay_determinism: ok")
    print("  epoch_plus_one_replay_determinism: ok")


if __name__ == "__main__":
    main()
