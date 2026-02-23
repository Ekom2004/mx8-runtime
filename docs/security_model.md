# MX8 Security Model (v1.8)

This document defines MX8 v1.8 security posture, assumptions, and hardening requirements.

## Current Security Posture (Shipped)

Control plane:

- `mx8-coordinator` and `mx8d-agent` communicate over gRPC.
- v1.8 does not ship built-in authn/authz or mTLS in coordinator/agent binaries.
- Access control is network-perimeter based today.

Data plane:

- Agents/processes fetch dataset bytes directly from storage (for example S3).
- Coordinator only serves control-plane APIs and manifest proxy bytes.
- Coordinator does not proxy dataset data bytes.

Secrets:

- MX8 uses ambient AWS SDK credential resolution for S3 clients.
- Common deployments rely on IAM role/workload identity or environment credentials.
- Do not store credentials in code or committed files.

## Trust Boundaries

1. Operator/workload boundary:
   - job config and environment variables supplied by trusted operator systems.
2. Control-plane boundary:
   - coordinator endpoint must only be reachable by trusted agents/operators.
3. Storage boundary:
   - dataset buckets/prefixes and manifest store backends must enforce least privilege.

## Authn/Authz Scope in v1.8

What is implemented:

- None in-process (no token verification or RBAC in coordinator API handlers).

What is required operationally:

- private network segmentation (VPC/subnet/ACL/SG/firewall)
- coordinator endpoint not publicly exposed
- per-job endpoint or strict ingress policy
- IAM policy separation for:
  - dataset read paths
  - manifest store read/write paths

## Recommended IAM Pattern (S3-backed deployments)

- Coordinator identity:
  - read/write manifest store prefix
  - read dataset metadata/listing needed for snapshot/index
- Agent/runtime identity:
  - read dataset objects
  - no manifest store write permissions required by default architecture

Use separate IAM principals where possible; avoid broad wildcard bucket permissions.

## Secret Handling Requirements

- Prefer workload identity/role-based auth over static keys.
- If environment credentials are used:
  - rotate regularly
  - scope minimally
  - avoid shell history/log leakage
- Mask secret values in logs and incident artifacts.

## Network Hardening Checklist

- [ ] Coordinator bind address scoped to private interfaces.
- [ ] Coordinator ingress restricted to known node CIDRs/security groups.
- [ ] No public internet exposure for coordinator port.
- [ ] Node egress to storage endpoints constrained to required hosts.
- [ ] mTLS/service mesh termination enabled if organization policy requires transport auth.

## Storage Hardening Checklist

- [ ] Separate buckets/prefixes for raw data vs manifest store.
- [ ] Least-privilege IAM policies on each principal.
- [ ] Object versioning/lifecycle policy applied to manifest store as required.
- [ ] Server-side encryption and key management aligned with org standards.

## Runtime Hardening Checklist

- [ ] Set production RSS cap (`MX8_MAX_PROCESS_RSS_BYTES`) by policy.
- [ ] Pin dataset links (`@sha256:...`) for reproducible reruns.
- [ ] Retain proof/metrics logs for audit and incident replay.
- [ ] Run deterministic prod gates before release (`./scripts/prod_readiness.sh`).

## Incident Response Minimums

For stale heartbeat, lease stall, manifest failures, and RSS breach procedures, use:

- `docs/prod_runbook.md`

## Known Security Gaps (v1.8)

1. No first-class coordinator API authn/authz.
2. No built-in TLS/mTLS configuration surface.
3. No fine-grained API-level authorization policy in coordinator.

These are deployment responsibilities until a dedicated security control-plane contract is shipped.
