# Source Resolver Protocol

This document defines the remote adapter protocol used when MX8 receives a non-native
dataset scheme and `MX8_SOURCE_RESOLVER_URL` is configured.

Native resolvers still run in-process:
- `s3://`
- local filesystem (`file://` or plain local path)
- `hf://`

For unknown schemes (for example `acme://...`), MX8 calls:

`POST <MX8_SOURCE_RESOLVER_URL>/resolve`

with JSON:

```json
{
  "link": "acme://team/dataset",
  "refresh": true,
  "recursive": true
}
```

## Auth

If `MX8_SOURCE_RESOLVER_AUTH_BEARER` is set, MX8 adds:

`Authorization: Bearer <token>`

to resolver requests.

## Response Contract

The resolver must return one of:

1. `manifest_tsv` (canonical TSV content), or
2. `records` (array of manifest rows)

Optional `manifest_hash` can be included for diagnostics. MX8 always recomputes
the hash locally and rejects mismatches.

Example response (records form):

```json
{
  "manifest_hash": "sha256:9f...",
  "records": [
    {
      "sample_id": 0,
      "location": "https://source.example.com/object/0",
      "byte_offset": null,
      "byte_length": null,
      "decode_hint": null
    }
  ]
}
```

Record rules enforced by MX8:
- valid `location`
- optional `byte_offset`/`byte_length` follow manifest invariants
- `sample_id` must be sequential from `0..N`

## Reliability

MX8 retries remote resolver calls with exponential backoff.

Controls:
- `MX8_SOURCE_RESOLVER_TIMEOUT_MS` (default `5000`)
- `MX8_SOURCE_RESOLVER_MAX_ATTEMPTS` (default `3`)

Transient statuses retried: `408`, `429`, `5xx`.

## Outcome

This lets operators attach new sources behind a resolver service without changing
MX8 runtime code. Users keep the same API: `mx8.load("<source-link>")`.
