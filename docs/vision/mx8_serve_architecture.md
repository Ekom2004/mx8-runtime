# Vision: MX8 Serve Architecture (The Native CDN Node)

## The Concept

MX8 currently consists of a highly optimized **Client** (`mx8.image`) and a globally distributed **Brain** (The Coordinator). To complete the pipeline and achieve total end-to-end performance, MX8 requires a native **Sender**.

`mx8.export()` (or `mx8 serve` via CLI) is a zero-configuration, purpose-built Rust web server embedded directly inside the MX8 binary. It transforms any directory of local hard drives into an internet-routable, MX8-optimized high-performance stream. It effectively allows any enterprise to build their own private, high-speed S3 alternative in seconds.

## Core Capabilities

### 1. The Dynamic Auto-Manifester (Zero "Data Prep")
Currently, users must run offline scripts to calculate byte offsets and generate a `manifest.tsv` before training.
With `mx8 serve`:
- The Rust server immediately scans the target directory upon launch.
- It calculates file sizes and builds a **Virtual Manifest** in RAM.
- When an MX8 worker connects to `http://<server-ip>/manifest`, it is instantly served the dynamically generated TSV. 
**Result:** The "Data Preparation" phase is eliminated. Users simply place files in a folder and begin training.

### 2. Physical OS Bypass (`sendfile` Optimization)
Traditional web servers copy data from the disk, up into the kernel, into user-space memory, and back down to the network card.
With `mx8 serve`:
- The server uses Rust's `tokio` and `axum` frameworks to intercept incoming `Range` requests from GPU workers.
- It leverages the Linux `sendfile()` syscall to instruct the kernel to pipe bytes directly from the hard drive platter to the physical Network Interface Card (NIC).
**Result:** The host CPU does effectively 0% of the data-transfer work, allowing a single desktop machine to saturate 100 Gbps network links to starving cloud GPUs.

### 3. The "Invisible" Local Stream
If a user writes: `loader = mx8.image("/home/user/local_data")`
- They do not use standard POSIX file reading.
- MX8 silently spins up an ephemeral `mx8.serve` instance bound to an internal UNIX domain socket (or `localhost`).
- The Python client requests bytes over this internal HTTP-like stream.
**Result:** The underlying execution path (byte-ranges, zero-copy, autotuning) remains 100% identical regardless of whether the data is sitting on the same motherboard or across the Atlantic Ocean.

## Proposed API

### The Python Interface (For ML Engineers)

To securely expose data to a GPU swarm across the open internet:

```python
import mx8

# This completely locks the process and begins serving bytes at line-rate.
mx8.export(
    source_dir="/mnt/massive_storage/dataset_v4",
    port=8080,
    auth="my_secure_token" # Required for public IP exposure
)
```

Cloud GPUs connect to the server perfectly seamlessly:

```python
loader = mx8.image(
    "http://203.0.113.42:8080/",
    auth="my_secure_token",
    batch=256,
    ram_gb=16.0,
    coord="cloud.mx8.dev"
)
```

### The CLI Interface (For IT / Sysadmins)

```bash
$ mx8 serve /mnt/nas/video_dataset --port 8080 --auth "corp_secret"

[MX8 Edge] Indexed 12,400,102 files in 3.1s
[MX8 Edge] Virtual Manifest constructed in RAM.
[MX8 Edge] Listening on 0.0.0.0:8080
[MX8 Edge] Ready to saturate network.
```

## Strategic Value

`mx8 serve` allows MX8 to capture the entire data plane. By removing the dependency on third-party web servers (NGINX) or heavy distributed filesystems (MinIO/S3), it allows companies to utilize their "dark data" on-premise without paying cloud storage or egress fees, effectively turning "local storage" into a native ML architecture.
