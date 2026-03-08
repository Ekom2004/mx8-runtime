# Vision: eBPF Telemetry & Autotuning

## The Concept

The MX8 Autotuner presently operates as an application-layer **Reactive** feedforward+PID controller. By limiting the `want` in-flight HTTP requests, MX8 gracefully avoids TCP slow-start and throttles buffer bloat. However, if the underlying network physically congests, the OS network stack still drops a packet, incurs a latency spike (100+ ms) while re-requesting the TCP fragment, and then the User-Space Rust app eventually corrects itself by turning the `want` dial down. 

With eBPF (Extended Berkeley Packet Filter), MX8 can transition from **Reactive Healing** to **Proactive Evasion**.

By injecting highly secure, JIT-compiled C/Rust programs into the absolute lowest levels of the Linux kernel network stack, MX8 can measure packet arrival variance at the nanosecond scale, communicating to the application-layer Autotuner *before* the application ever experiences congestion.

## Core Capabilities

### 1. Telepathic Autotuning
The eBPF module sits directly inside the host's Linux TCP stack, listening for incoming data. 
- It tracks the exact timestamp each raw 1,500-byte packet arrives from Cloudflare or S3.
- If the RTT (Round Trip Time) or inter-packet gap stretches by single milliseconds, it identifies that an upstream router is filling up.
- The eBPF program writes a `CONGESTION_IMMINENT` signal into an eBPF Shared Memory Map.
- **Result:** The `mx8-py/src/autotune.rs` PID controller polls this map 1,000 times a second and lowers the `want` throttle before a packet actually drops, maintaining an unbreakable, flawlessly flat, mathematically optimized throughput line.

### 2. The AF_XDP (Zero-Copy Receive) Paradigm
While MX8 already employs Zero-Copy *Decoding* via `device_output=True` (shipping compressed bytes to the NVDEC unzipper), the Operating System still performs a massive memcpy when translating the network payload from the Network Interface Card (NIC) to the User-Space Rust memory map.

Using AF_XDP:
- MX8 allocates a large contiguous RAM buffer on boot.
- The eBPF module hooks directly onto the network card driver.
- When an HTTP byte-range frame arrives matching the MX8 socket, the NIC performs a Direct Memory Access (DMA) write, dropping the raw packet array straight into the Rust application buffer.
- **Result:** The host CPU does effectively 0% of the network copy work, unlocking 100 Gbps to 400 Gbps network saturation for a single server processor.

## Architecture

This is fully implementable in native Rust using libraries like `aya`.

### The Sub-Kernel Probe (eBPF Rust)
```rust
#[tracepoint(name = "tcp_rcv_established")]
pub fn measure_tcp_spacing(ctx: TracePointContext) -> u32 {
    let packet_time = bpf_ktime_get_ns();
    let old_time = read_from_map(socket_id);
    let gap = packet_time - old_time;
    
    // Alert the user-space PID loop of micro-congestion
    if gap > THRESHOLD {
        write_alert_to_shared_memory(socket_id, CONGESTION_WARNING);
    }
    save_to_map(socket_id, packet_time);
    return 0;
}
```

### The Tying (User-Space Link)

```rust
// Inside `mx8-py/src/autotune.rs`
let network_status = bpf_map.read(current_socket_id);

if network_status == CONGESTION_WARNING {
    // TCP queue expanding; back off request limit by 1
    next.want = next.want.saturating_sub(1);
} else {
    // Network is clear, continue throttling up
    next.want = next.want.saturating_add(1);
}
```

## Graceful Degradation
Because eBPF is a deeply privileged Linux-only technology, MX8 treats it purely as an **Accelerator**. If an ML engineer runs MX8 on macOS or Windows (or a Linux instance without `root`/`CAP_BPF` permissions), the eBPF hook quietly disables itself. The standard `autotune.rs` pipeline falls back to Application-Layer application-time sampling metrics seamlessly, guaranteeing the pipeline runs perfectly regardless of OS privilege.
