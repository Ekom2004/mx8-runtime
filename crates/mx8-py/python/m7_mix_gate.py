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
    on_source_exhausted: str = "allow",
    mix_profile: str = "balanced",
    mix_autotune: bool = True,
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
        on_source_exhausted=on_source_exhausted,
        profile=mix_profile,
        autotune=mix_autotune,
        constraints=mx8.Constraints(
            max_inflight_bytes=max_inflight_bytes,
            max_process_rss_bytes=max_process_rss_bytes,
        ),
        runtime=mx8.RuntimeConfig(
            prefetch_batches=1,
            max_queue_batches=8,
        ),
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


def _run_source_exhaustion_policy_checks(
    *,
    store_root: str,
    link_a: str,
    link_b: str,
    manifest_a: str,
    manifest_b: str,
    max_inflight_bytes: int,
    max_process_rss_bytes: int,
) -> None:
    loader_a = mx8.load(
        link_a,
        manifest_store_root=store_root,
        dev_manifest_path=manifest_a,
        batch_size_samples=1,
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
        constraints=mx8.Constraints(
            max_inflight_bytes=max_inflight_bytes,
            max_process_rss_bytes=max_process_rss_bytes,
        ),
    )
    mixed_error = mx8.mix(
        [loader_a, loader_b],
        weights=[1.0, 1.0],
        seed=19,
        epoch=5,
        on_source_exhausted="error",
        profile="balanced",
        autotune=True,
        constraints=mx8.Constraints(
            max_inflight_bytes=max_inflight_bytes,
            max_process_rss_bytes=max_process_rss_bytes,
        ),
        runtime=mx8.RuntimeConfig(
            prefetch_batches=1,
            max_queue_batches=8,
        ),
    )
    seen = 0
    try:
        for _batch in mixed_error:
            seen += 1
    except RuntimeError as err:
        message = str(err)
        if "source exhausted" not in message:
            raise RuntimeError(f"unexpected exhaustion error message: {message}") from err
    else:
        raise RuntimeError("expected RuntimeError for on_source_exhausted=error")
    if seen == 0:
        raise RuntimeError("source exhaustion error fired before any progress")

    loader_a_allow = mx8.load(
        link_a,
        manifest_store_root=store_root,
        dev_manifest_path=manifest_a,
        batch_size_samples=1,
        constraints=mx8.Constraints(
            max_inflight_bytes=max_inflight_bytes,
            max_process_rss_bytes=max_process_rss_bytes,
        ),
    )
    loader_b_allow = mx8.load(
        link_b,
        manifest_store_root=store_root,
        dev_manifest_path=manifest_b,
        batch_size_samples=1,
        constraints=mx8.Constraints(
            max_inflight_bytes=max_inflight_bytes,
            max_process_rss_bytes=max_process_rss_bytes,
        ),
    )
    mixed_allow = mx8.mix(
        [loader_a_allow, loader_b_allow],
        weights=[1.0, 1.0],
        seed=19,
        epoch=5,
        on_source_exhausted="allow",
        profile="balanced",
        autotune=True,
        constraints=mx8.Constraints(
            max_inflight_bytes=max_inflight_bytes,
            max_process_rss_bytes=max_process_rss_bytes,
        ),
        runtime=mx8.RuntimeConfig(
            prefetch_batches=1,
            max_queue_batches=8,
        ),
    )
    seen_allow = 0
    for _batch in mixed_allow:
        seen_allow += 1
    if seen_allow == 0:
        raise RuntimeError("expected progress for on_source_exhausted=allow")
    allow_stats = mixed_allow.stats()
    policy = str(allow_stats.get("mix_source_exhaustion_policy", ""))
    if policy != "allow":
        raise RuntimeError(f"unexpected allow policy stat: {policy}")
    exhausted_total = list(allow_stats.get("mix_source_exhausted_total", []))
    if len(exhausted_total) != 2:
        raise RuntimeError(f"unexpected exhausted_total shape: {exhausted_total}")
    if sum(int(v) for v in exhausted_total) < 2:
        raise RuntimeError(
            f"expected exhausted counters for both sources, got={exhausted_total}"
        )

    try:
        mx8.mix(
            [loader_a_allow, loader_b_allow],
            weights=[1.0, 1.0],
            runtime=mx8.RuntimeConfig(want=2),
        )
    except ValueError:
        pass
    else:
        raise RuntimeError("expected ValueError for mix runtime.want override")


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
    strict_mode = os.environ.get("MX8_MIX_GATE_STRICT", "0") == "1"
    expect_epoch_drift = os.environ.get(
        "MX8_MIX_GATE_EXPECT_EPOCH_DRIFT",
        "1" if strict_mode else "0",
    ) == "1"
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
        on_source_exhausted="allow",
        mix_profile="balanced",
        mix_autotune=True,
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
        on_source_exhausted="allow",
        mix_profile="balanced",
        mix_autotune=True,
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
        on_source_exhausted="allow",
        mix_profile="balanced",
        mix_autotune=True,
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
        on_source_exhausted="allow",
        mix_profile="balanced",
        mix_autotune=True,
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
        on_source_exhausted="allow",
        mix_profile="balanced",
        mix_autotune=True,
    )
    digest_epoch_plus_one_a = _digest_seq(seq_epoch_plus_one_a)
    digest_epoch_plus_one_b = _digest_seq(seq_epoch_plus_one_b)
    if digest_epoch_plus_one_a != digest_epoch_plus_one_b:
        raise RuntimeError("mix replay mismatch for repeated epoch+1 run")
    if expect_epoch_drift and digest_epoch_plus_one_a == digest1:
        raise RuntimeError(
            "mix epoch drift expectation failed: epoch+1 digest matched base epoch digest"
        )

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

    exhaustion_policy = str(stats1.get("mix_source_exhaustion_policy", ""))
    if exhaustion_policy != "allow":
        raise RuntimeError(
            f"unexpected source exhaustion policy in stats: {exhaustion_policy}"
        )
    exhausted_total = list(stats1.get("mix_source_exhausted_total", []))
    if len(exhausted_total) != 2:
        raise RuntimeError(f"unexpected mix_source_exhausted_total shape: {exhausted_total}")
    if any(int(v) != 0 for v in exhausted_total):
        raise RuntimeError(
            f"mix_source_exhausted_total should remain zero in non-exhaustion run: {exhausted_total}"
        )

    mix_profile = str(stats1.get("mix_profile", ""))
    if mix_profile != "balanced":
        raise RuntimeError(f"unexpected mix profile: {mix_profile}")
    if not bool(stats1.get("mix_autotune_enabled", False)):
        raise RuntimeError("mix autotune expected enabled")
    if int(stats1.get("mix_effective_prefetch_batches", 0)) != 1:
        raise RuntimeError("mix_effective_prefetch_batches expected 1")
    if int(stats1.get("mix_effective_max_queue_batches", 0)) != 8:
        raise RuntimeError("mix_effective_max_queue_batches expected 8")
    source_diag = list(stats1.get("mix_sources", []))
    if len(source_diag) != 2:
        raise RuntimeError(f"expected two mix source diagnostics, got={len(source_diag)}")
    for idx, src in enumerate(source_diag):
        if int(src.get("source_idx", -1)) != idx:
            raise RuntimeError(f"unexpected source_idx in diagnostics: {src}")
        if int(src.get("prefetch_batches", 0)) != 1:
            raise RuntimeError(f"source {idx} prefetch override missing: {src}")
        if int(src.get("max_queue_batches", 0)) != 8:
            raise RuntimeError(f"source {idx} queue override missing: {src}")
        metrics = src.get("metrics", {})
        if "process_rss_bytes" not in metrics:
            raise RuntimeError(f"source {idx} metrics missing process_rss_bytes")

    if os.environ.get("MX8_MIX_GATE_CHECK_EXHAUSTION", "1") == "1":
        _run_source_exhaustion_policy_checks(
            store_root=store_root,
            link_a=link_a,
            link_b=link_b,
            manifest_a=manifest_a,
            manifest_b=manifest_b,
            max_inflight_bytes=max_inflight_bytes,
            max_process_rss_bytes=max_process_rss_bytes,
        )

    print("mix_gate_summary")
    print("  strict_mode:", strict_mode)
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
    print("  source_exhaustion_policy_allow_stats: ok")
    if os.environ.get("MX8_MIX_GATE_CHECK_EXHAUSTION", "1") == "1":
        print("  source_exhaustion_policy_error_vs_allow: ok")
    if expect_epoch_drift:
        print("  epoch_drift_digest_change: ok")


if __name__ == "__main__":
    main()
