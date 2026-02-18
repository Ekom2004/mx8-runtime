from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Rails:
    min_prefetch: int
    max_prefetch: int
    min_queue: int
    max_queue: int
    min_want: int
    max_want: int


@dataclass
class State:
    wait_ewma: float = 0.0
    rss_ewma: float = 0.0
    prev_rss_ewma: float = 0.0
    integral_rss: float = 0.0
    cooldown_ticks: int = 0
    increase_ticks: int = 0


@dataclass
class Knobs:
    prefetch: int
    max_queue: int
    want: int


def clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def tick(
    state: State,
    current: Knobs,
    rails: Rails,
    wait_ratio: float,
    rss_ratio: float,
    inflight_ratio: float,
    dt: float = 2.0,
) -> Tuple[Knobs, str, float]:
    alpha = 0.2
    wait_target = 0.05
    rss_target = 0.90
    hard_cut_rss = 0.97
    hard_cut_inflight = 0.98
    soft_cut_pressure = 0.60
    increase_pressure_gate = 0.30
    kp, ki, kd = 1.5, 0.2, 0.1

    state.wait_ewma = alpha * wait_ratio + (1.0 - alpha) * state.wait_ewma
    state.rss_ewma = alpha * rss_ratio + (1.0 - alpha) * state.rss_ewma
    e = state.rss_ewma - rss_target
    state.integral_rss = clamp(state.integral_rss + e * dt, -0.5, 0.5)
    d = (state.rss_ewma - state.prev_rss_ewma) / dt
    state.prev_rss_ewma = state.rss_ewma

    pressure = clamp(kp * e + ki * state.integral_rss + kd * d, 0.0, 1.0)
    next_knobs = Knobs(current.prefetch, current.max_queue, current.want)
    reason = "hold"

    hard_cut = rss_ratio >= hard_cut_rss or inflight_ratio >= hard_cut_inflight
    if hard_cut:
        next_knobs.prefetch = max(rails.min_prefetch, int(next_knobs.prefetch * 0.5))
        next_knobs.max_queue = max(rails.min_queue, int(next_knobs.max_queue * 0.7))
        next_knobs.want = max(rails.min_want, int(next_knobs.want * 0.5))
        state.cooldown_ticks = 2
        state.increase_ticks = 0
        reason = "hard_cut"
    elif pressure >= soft_cut_pressure:
        next_knobs.prefetch = max(rails.min_prefetch, next_knobs.prefetch - 1)
        next_knobs.max_queue = max(rails.min_queue, next_knobs.max_queue - 2)
        reason = "soft_cut"
        state.increase_ticks = 0
        if state.cooldown_ticks > 0:
            state.cooldown_ticks -= 1
    else:
        starvation = state.wait_ewma > wait_target
        if (
            state.cooldown_ticks == 0
            and starvation
            and pressure <= increase_pressure_gate
            and inflight_ratio <= 0.85
        ):
            next_knobs.prefetch = min(rails.max_prefetch, next_knobs.prefetch + 1)
            next_knobs.max_queue = min(rails.max_queue, next_knobs.max_queue + 2)
            state.increase_ticks += 1
            if state.increase_ticks % 2 == 0:
                next_knobs.want = min(rails.max_want, next_knobs.want + 1)
            reason = "aimd_increase"
        else:
            state.increase_ticks = 0
            if state.cooldown_ticks > 0:
                state.cooldown_ticks -= 1

    return next_knobs, reason, pressure


def main() -> None:
    rails = Rails(
        min_prefetch=2,
        max_prefetch=16,
        min_queue=32,
        max_queue=256,
        min_want=1,
        max_want=8,
    )
    state = State()
    knobs = Knobs(prefetch=8, max_queue=96, want=4)

    # (phase, wait_ratio, rss_ratio, inflight_ratio)
    # warmup -> pressure spike -> sustained pressure -> recovery
    signal: List[Tuple[str, float, float, float]] = []
    signal += [("warmup", 0.08, 0.45, 0.40)] * 3
    signal += [("pressure", 0.02, 0.98, 0.55)] * 3
    signal += [("pressure", 0.03, 0.95, 0.55)] * 3
    signal += [("recovery", 0.16, 0.55, 0.35)] * 5

    saw_cut = False
    saw_increase = False
    for i, (phase, wait_ratio, rss_ratio, inflight_ratio) in enumerate(signal):
        next_knobs, reason, pressure = tick(
            state, knobs, rails, wait_ratio, rss_ratio, inflight_ratio
        )
        if reason in ("hard_cut", "soft_cut"):
            saw_cut = True
        if reason == "aimd_increase":
            saw_increase = True
        print(
            f"tick={i:02d} phase={phase} wait={wait_ratio:.2f} rss={rss_ratio:.2f} "
            f"inflight={inflight_ratio:.2f} pressure={pressure:.3f} "
            f"reason={reason} knobs=p{next_knobs.prefetch}/q{next_knobs.max_queue}/w{next_knobs.want}"
        )
        knobs = next_knobs

    if not saw_cut:
        raise SystemExit("simulation failed: no cut observed under pressure")
    if not saw_increase:
        raise SystemExit("simulation failed: no recovery increase observed")

    print(
        "autotune_memory_pressure_sim_summary "
        f"saw_cut={int(saw_cut)} saw_increase={int(saw_increase)} "
        f"final_knobs=p{knobs.prefetch}/q{knobs.max_queue}/w{knobs.want}"
    )


if __name__ == "__main__":
    main()
