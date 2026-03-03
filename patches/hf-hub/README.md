# hf-hub (local patch)

Local workspace patch for `hf-hub` used by MX8 wheel builds.

Patch intent:
- avoid `native-tls` / OpenSSL transitive dependencies in cross Linux builds
- keep behavior compatible with upstream `hf-hub` v0.3.2 for MX8 usage
