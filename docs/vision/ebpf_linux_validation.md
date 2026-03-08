# eBPF Net-Pressure Linux Validation

This runbook validates Phase 1 net-pressure integration on a Linux host.

## Prereqs

- Linux host (Ubuntu/RHEL/etc)
- MX8 build with `mx8_ebpf_net_pressure` feature
- Networked dataset workload (HTTP/S3/GCS path) that can produce sustained traffic

## 1. Enable source

```bash
export MX8_AUTOTUNE_NET_SOURCE=ebpf
```

## 2. Run workload

Use a normal MX8 loader call (no API changes required). Example:

```python
import mx8

ldr = mx8.load("http://your-source@refresh", batch=512, ram_gb=24)
for i, _ in enumerate(ldr):
    if i % 50 == 0:
        s = ldr.stats()
        print(
            i,
            s.get("autotune_net_pressure_ratio"),
            s.get("autotune_net_signal_age_ms"),
            s.get("autotune_net_signal_stale_total"),
            s.get("autotune_net_assisted_backoff_total"),
            s.get("autotune_net_disabled_total"),
        )
```

## 3. Expected signals

- `autotune_net_disabled_total == 0` when source initializes correctly
- `autotune_net_signal_age_ms` remains bounded and updates over time
- `autotune_net_pressure_ratio` in `[0.0, 1.0]`
- `autotune_net_signal_stale_total` stays low in healthy collection
- `autotune_net_assisted_backoff_total` increments under induced network pressure

## 4. Fallback behavior checks

If the source cannot initialize, runtime must fail open:

- `autotune_net_disabled_total > 0`
- `autotune_net_pressure_ratio == 0.0`
- loader continues operating with existing autotune path

## 5. Suggested pressure test

- Introduce controlled packet loss/latency with `tc netem`
- Compare baseline vs pressured run:
  - throughput variance
  - hard-cut count
  - `autotune_net_assisted_backoff_total`

