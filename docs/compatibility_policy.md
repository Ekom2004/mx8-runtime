# MX8 Compatibility Policy (v1.8)

This policy defines stability levels and compatibility guarantees for MX8 public surfaces.

## Version Axes

1. Product/runtime docs version:
   - current contract target: `v1.8`
2. gRPC namespace version:
   - current package: `mx8.v0`
3. Manifest schema version:
   - current constant: `MANIFEST_SCHEMA_VERSION = 0`

These axes are independent and must be interpreted separately.

## Stability Levels

- `stable`: contractual surface; backward-compatibility expected within a major line.
- `experimental`: shipped but allowed to evolve with smaller notice and fewer guarantees.
- `internal`: not part of public contract; can change without compatibility guarantees.

## Surface-by-Surface Policy

### 1) Dataset link grammar (`stable`)

Supported forms:

- plain: `s3://bucket/prefix/` or `/path`
- refresh: `...@refresh`
- pinned: `...@sha256:<manifest_hash>`

Policy:

- existing forms remain valid within `v1.x`.
- semantic behavior (pinned vs refresh) remains consistent.

### 2) Canonical manifest format (`stable`)

Policy:

- `schema_version=0` canonical TSV stays readable across `v1.x`.
- writers may add fields only through documented schema-version bumps.
- any incompatible reader/writer change requires a new schema version and migration notes.

### 3) gRPC wire contract (`stable` for `mx8.v0`)

Policy:

- protobuf field numbers are immutable once released.
- compatible changes in `mx8.v0` are additive only:
  - add new optional fields
  - add new RPCs
- incompatible changes require a new namespace version (for example `mx8.v1`).

### 4) Python API (`stable` core, `experimental` selected)

Stable:

- documented constructor args and behavior in `docs/python_api.md`.

Experimental:

- clearly labeled feature-path knobs and fast-evolving runtime controls.

Internal:

- `mx8._internal.*` is internal and not compatibility-guaranteed.

### 5) Stats surface (`experimental`)

Policy:

- `loader.stats()` keys may expand over time.
- field existence/types for non-core diagnostic keys are not yet frozen.
- operator tooling should tolerate additive keys and missing optional fields.

### 6) CLI tools (`stable` command identity, additive args)

Policy:

- command names remain stable:
  - `mx8-pack-s3`
  - `mx8-snapshot-resolve`
  - `mx8-seed-s3`
- existing args/env mappings remain valid within `v1.x`.
- new flags may be added additively.

## Deprecation Window

For `stable` surfaces:

- minimum deprecation window: 2 minor releases before removal, or explicit major-version boundary.
- deprecations must be documented in release notes/docs before removal.

For `experimental` surfaces:

- may change in any minor release with docs updates.
- no fixed multi-release deprecation guarantee.

For `internal` surfaces:

- no compatibility guarantees.

## Change Management Requirements

When changing a `stable` surface:

1. update the owning contract doc in the same change
2. provide migration guidance
3. add/adjust gate coverage for the changed invariant

## Current Known Inconsistency to Resolve

- Repo-level version labeling is not yet fully aligned across all top-level docs.
- Run `DOC-201` from `docs/documentation_backlog.md` to normalize all version markers.
