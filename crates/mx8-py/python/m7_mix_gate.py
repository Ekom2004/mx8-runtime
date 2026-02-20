import os
from hashlib import blake2b

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
    weights: list[float],
    max_inflight_bytes: int,
    max_process_rss_bytes: int,
):
    loader_a = mx8.load(
        link_a,
        manifest_store_root=store_root,
        dev_manifest_path=manifest_a,
        batch_size_samples=1,
        max_queue_batches=8,
        prefetch_batches=1,
        constraints=mx8.Constraints(
            max_inflight_bytes=max_inflight_bytes,
            max_process_rss_bytes=max_process_rss_bytes,
        ),
    )
    loader_b = mx8.load(
        link_b,
        manifest_store_root=store_root,
        dev_manifest_path=manifest_b,
        batch_size_samples=1,
        max_queue_batches=8,
        prefetch_batches=1,
        constraints=mx8.Constraints(
            max_inflight_bytes=max_inflight_bytes,
            max_process_rss_bytes=max_process_rss_bytes,
        ),
    )

    mixed = mx8.mix(
        [loader_a, loader_b],
        weights=weights,
        seed=seed,
        epoch=epoch,
        starvation_window=10_000,
    )

    seq: list[tuple[str, int]] = []
    max_total_inflight = 0
    max_process_rss = 0
    for _, batch in zip(range(steps), mixed):
        sample_ids = batch.sample_ids
        if len(sample_ids) != 1:
            raise RuntimeError(f"expected batch_size_samples=1, got={len(sample_ids)}")
        seq.append((_source_tag(batch), int(sample_ids[0])))

        stats = mixed.stats()
        total_inflight = int(stats.get("mix_total_inflight_bytes", 0))
        shared_max = int(stats.get("mix_shared_max_inflight_bytes", 0))
        if total_inflight > shared_max:
            raise RuntimeError(
                "mix shared inflight cap breached during run: "
                f"total_inflight={total_inflight} shared_max={shared_max}"
            )
        max_total_inflight = max(max_total_inflight, total_inflight)

        loader_a_rss = int(loader_a.stats().get("process_rss_bytes", 0))
        loader_b_rss = int(loader_b.stats().get("process_rss_bytes", 0))
        max_process_rss = max(max_process_rss, loader_a_rss, loader_b_rss)
        if loader_a_rss > max_process_rss_bytes or loader_b_rss > max_process_rss_bytes:
            raise RuntimeError(
                "mix process RSS cap breached during run: "
                f"loader_a_rss={loader_a_rss} loader_b_rss={loader_b_rss} "
                f"max_process_rss_bytes={max_process_rss_bytes}"
            )

    stats = mixed.stats()
    return seq, stats, max_total_inflight, max_process_rss


def _digest_seq(seq: list[tuple[str, int]]) -> str:
    h = blake2b(digest_size=16)
    for source_tag, sample_id in seq:
        h.update(source_tag.encode("utf-8"))
        h.update(b":")
        h.update(str(sample_id).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main() -> None:
    store_root = os.environ["MX8_MANIFEST_STORE_ROOT"]
    link_a = os.environ["MX8_DATASET_LINK_A"]
    link_b = os.environ["MX8_DATASET_LINK_B"]
    manifest_a = os.environ["MX8_DEV_MANIFEST_PATH_A"]
    manifest_b = os.environ["MX8_DEV_MANIFEST_PATH_B"]

    steps = int(os.environ.get("MX8_MIX_GATE_STEPS", "200"))
    seed = int(os.environ.get("MX8_MIX_GATE_SEED", "17"))
    epoch = int(os.environ.get("MX8_MIX_GATE_EPOCH", "3"))
    ratio_tol = float(os.environ.get("MX8_MIX_GATE_RATIO_TOL", "0.02"))
    max_inflight_bytes = int(os.environ.get("MX8_MIX_GATE_MAX_INFLIGHT_BYTES", str(8 * 1024 * 1024)))
    max_process_rss_bytes = int(
        os.environ.get("MX8_MIX_GATE_MAX_PROCESS_RSS_BYTES", str(2 * 1024 * 1024 * 1024))
    )
    raw_weights = os.environ.get("MX8_MIX_GATE_WEIGHTS", "0.7,0.3")
    try:
        weights = [float(part.strip()) for part in raw_weights.split(",")]
    except ValueError as err:
        raise RuntimeError(f"invalid MX8_MIX_GATE_WEIGHTS={raw_weights!r}") from err
    if len(weights) != 2 or any(w <= 0.0 for w in weights):
        raise RuntimeError(f"MX8_MIX_GATE_WEIGHTS must be two positive weights, got={weights}")
    weight_sum = weights[0] + weights[1]
    target_ratio = [weights[0] / weight_sum, weights[1] / weight_sum]

    seq1, stats1, max_inflight_1, max_rss_1 = _run_once(
        store_root=store_root,
        link_a=link_a,
        link_b=link_b,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        seed=seed,
        epoch=epoch,
        steps=steps,
        weights=weights,
        max_inflight_bytes=max_inflight_bytes,
        max_process_rss_bytes=max_process_rss_bytes,
    )
    seq2, _stats2, max_inflight_2, max_rss_2 = _run_once(
        store_root=store_root,
        link_a=link_a,
        link_b=link_b,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        seed=seed,
        epoch=epoch,
        steps=steps,
        weights=weights,
        max_inflight_bytes=max_inflight_bytes,
        max_process_rss_bytes=max_process_rss_bytes,
    )
    seq3, _stats3, max_inflight_3, max_rss_3 = _run_once(
        store_root=store_root,
        link_a=link_a,
        link_b=link_b,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        seed=seed,
        epoch=epoch,
        steps=steps,
        weights=weights,
        max_inflight_bytes=max_inflight_bytes,
        max_process_rss_bytes=max_process_rss_bytes,
    )
    digest1 = _digest_seq(seq1)
    digest2 = _digest_seq(seq2)
    digest3 = _digest_seq(seq3)
    if not (digest1 == digest2 == digest3):
        raise RuntimeError(
            "mix replay mismatch for identical seed/epoch runs: "
            f"{digest1} {digest2} {digest3}"
        )

    seq_epoch_plus_one_a, _stats4a, _max_inflight_4a, _max_rss_4a = _run_once(
        store_root=store_root,
        link_a=link_a,
        link_b=link_b,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        seed=seed,
        epoch=epoch + 1,
        steps=steps,
        weights=weights,
        max_inflight_bytes=max_inflight_bytes,
        max_process_rss_bytes=max_process_rss_bytes,
    )
    seq_epoch_plus_one_b, _stats4b, _max_inflight_4b, _max_rss_4b = _run_once(
        store_root=store_root,
        link_a=link_a,
        link_b=link_b,
        manifest_a=manifest_a,
        manifest_b=manifest_b,
        seed=seed,
        epoch=epoch + 1,
        steps=steps,
        weights=weights,
        max_inflight_bytes=max_inflight_bytes,
        max_process_rss_bytes=max_process_rss_bytes,
    )
    digest_epoch_plus_one_a = _digest_seq(seq_epoch_plus_one_a)
    digest_epoch_plus_one_b = _digest_seq(seq_epoch_plus_one_b)
    if digest_epoch_plus_one_a != digest_epoch_plus_one_b:
        raise RuntimeError("mix replay mismatch for repeated epoch+1 run")

    realized = list(stats1.get("mix_realized_ratio", []))
    if len(realized) != 2:
        raise RuntimeError(f"unexpected realized ratio shape: {realized}")
    if abs(realized[0] - target_ratio[0]) > ratio_tol or abs(realized[1] - target_ratio[1]) > ratio_tol:
        raise RuntimeError(
            "mix realized ratio outside tolerance: "
            f"got={realized} target={target_ratio} tol={ratio_tol}"
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
    shared_inflight_violations = int(stats1.get("mix_shared_inflight_violation_total", 0))
    if shared_inflight_violations != 0:
        raise RuntimeError(
            f"mix_shared_inflight_violation_total expected 0, got={shared_inflight_violations}"
        )
    if int(stats1.get("mix_total_inflight_bytes", 0)) > int(
        stats1.get("mix_shared_max_inflight_bytes", 0)
    ):
        raise RuntimeError("mix total inflight exceeds shared cap at end of run")

    max_observed_inflight = max(max_inflight_1, max_inflight_2, max_inflight_3)
    max_observed_rss = max(max_rss_1, max_rss_2, max_rss_3)
    if max_observed_rss > max_process_rss_bytes:
        raise RuntimeError(
            "mix observed process RSS above cap across runs: "
            f"max_observed_rss={max_observed_rss} max_process_rss_bytes={max_process_rss_bytes}"
        )

    print("mix_gate_summary")
    print("  steps:", steps)
    print("  weights:", weights)
    print("  target_ratio:", target_ratio)
    print("  realized_ratio:", realized)
    print("  starvation:", starvation)
    print("  ratio_tol:", ratio_tol)
    print("  max_observed_inflight_bytes:", max_observed_inflight)
    print("  max_observed_process_rss_bytes:", max_observed_rss)
    print("  configured_max_process_rss_bytes:", max_process_rss_bytes)
    print("  snapshot_period_ticks:", snapshot_period)
    print("  snapshot_emitted_total:", snapshot_emitted)
    print(
        "  sequence_head:",
        ",".join(f"{source}:{sample_id}" for source, sample_id in seq1[:12]),
    )
    print(
        "  determinism_digest_runs:",
        ",".join([digest1, digest2, digest3]),
    )
    print("  replay_determinism_runs=3: ok")
    print("  epoch_plus_one_replay_determinism: ok")


if __name__ == "__main__":
    main()
