use anyhow::Result;

#[cfg(feature = "azure")]
fn parse_boolish(key: &str, value: &str) -> Result<bool> {
    let s = value.trim().to_ascii_lowercase();
    match s.as_str() {
        "1" | "true" | "yes" | "y" | "on" => Ok(true),
        "0" | "false" | "no" | "n" | "off" => Ok(false),
        _ => anyhow::bail!(
            "invalid boolean env var {}={:?} (expected true/false/1/0)",
            key,
            value
        ),
    }
}

#[cfg(feature = "azure")]
fn parse_env_bool(key: &str) -> Result<Option<bool>> {
    match std::env::var(key) {
        Ok(v) => Ok(Some(parse_boolish(key, &v)?)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(anyhow::Error::new(e)),
    }
}

#[cfg(feature = "azure")]
fn parse_conn_str_field(conn_str: &str, field: &str) -> Option<String> {
    conn_str.split(';').find_map(|entry| {
        let (k, v) = entry.split_once('=')?;
        if k.trim().eq_ignore_ascii_case(field) {
            let value = v.trim();
            if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            }
        } else {
            None
        }
    })
}

/// Build an Azure Blob Storage `ClientBuilder` from env vars.
///
/// Auth resolution order:
///   1. `AZURE_STORAGE_CONNECTION_STRING` → connection string auth
///   2. `AZURE_STORAGE_ACCOUNT` + `AZURE_STORAGE_ACCESS_KEY` → access key auth
///
/// Override the storage endpoint via `MX8_AZURE_ENDPOINT_URL` (useful for
/// local emulators such as Azurite).
///
/// For local emulator-only validation, set `MX8_AZURE_ANONYMOUS=1` and
/// `MX8_AZURE_ENDPOINT_URL=<emulator_url>`. When anonymous, the account name
/// defaults to `devstoreaccount1` (Azurite default) unless
/// `AZURE_STORAGE_ACCOUNT` is set explicitly.
#[cfg(feature = "azure")]
pub fn client_builder_from_env() -> Result<azure_storage_blobs::prelude::ClientBuilder> {
    client_builder_from_env_with_endpoint_override(None)
}

/// Same as `client_builder_from_env` but allows an explicit endpoint override.
#[cfg(feature = "azure")]
pub fn client_builder_from_env_with_endpoint_override(
    endpoint_override: Option<&str>,
) -> Result<azure_storage_blobs::prelude::ClientBuilder> {
    use azure_storage::prelude::*;
    use azure_storage::CloudLocation;
    use azure_storage_blobs::prelude::*;

    let endpoint_url = endpoint_override
        .map(|s| s.to_string())
        .or_else(|| std::env::var("MX8_AZURE_ENDPOINT_URL").ok());
    let anonymous = parse_env_bool("MX8_AZURE_ANONYMOUS")?.unwrap_or(false);

    if anonymous {
        let endpoint_url = endpoint_url.ok_or_else(|| {
            anyhow::anyhow!(
                "MX8_AZURE_ANONYMOUS=1 requires MX8_AZURE_ENDPOINT_URL (emulator endpoint)"
            )
        })?;
        let account = std::env::var("AZURE_STORAGE_ACCOUNT")
            .unwrap_or_else(|_| "devstoreaccount1".to_string());
        let creds = StorageCredentials::anonymous();
        let mut builder = ClientBuilder::new(&account, creds);
        builder = builder.cloud_location(CloudLocation::Custom {
            account: account.clone(),
            uri: endpoint_url,
        });
        anyhow::ensure!(
            !account.trim().is_empty(),
            "AZURE_STORAGE_ACCOUNT must be non-empty when provided"
        );
        return Ok(builder);
    }

    // Try connection string first.
    if let Ok(conn_str) = std::env::var("AZURE_STORAGE_CONNECTION_STRING") {
        // Parse AccountName/AccountKey (+ BlobEndpoint if present) from connection string.
        let account = parse_conn_str_field(&conn_str, "AccountName").ok_or_else(|| {
            anyhow::anyhow!("AZURE_STORAGE_CONNECTION_STRING missing AccountName")
        })?;
        let key = parse_conn_str_field(&conn_str, "AccountKey")
            .ok_or_else(|| anyhow::anyhow!("AZURE_STORAGE_CONNECTION_STRING missing AccountKey"))?;
        let blob_endpoint = parse_conn_str_field(&conn_str, "BlobEndpoint");
        let creds = StorageCredentials::access_key(&account, key);
        let mut builder = ClientBuilder::new(&account, creds);
        if let Some(url) = endpoint_url.or(blob_endpoint) {
            builder = builder.cloud_location(CloudLocation::Custom {
                account: account.clone(),
                uri: url,
            });
        }
        return Ok(builder);
    }

    // Fall back to account + access key.
    let account = std::env::var("AZURE_STORAGE_ACCOUNT").map_err(|_| {
        anyhow::anyhow!(
            "Azure auth: set AZURE_STORAGE_CONNECTION_STRING or \
             (AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_ACCESS_KEY)"
        )
    })?;
    let access_key = std::env::var("AZURE_STORAGE_ACCESS_KEY").map_err(|_| {
        anyhow::anyhow!(
            "Azure auth: AZURE_STORAGE_ACCOUNT is set but AZURE_STORAGE_ACCESS_KEY is missing"
        )
    })?;

    let creds = StorageCredentials::access_key(&account, access_key);
    let mut builder = ClientBuilder::new(&account, creds);
    if let Some(url) = endpoint_url {
        builder = builder.cloud_location(CloudLocation::Custom {
            account: account.clone(),
            uri: url,
        });
    }
    Ok(builder)
}

#[cfg(all(test, feature = "azure"))]
mod tests {
    use super::{parse_boolish, parse_conn_str_field};

    #[test]
    fn parse_boolish_accepts_truthy_and_falsy_values() {
        for v in ["1", "true", "yes", "y", "on", " TRUE "] {
            assert!(parse_boolish("K", v).expect("truthy parse"));
        }
        for v in ["0", "false", "no", "n", "off", " OFF "] {
            assert!(!parse_boolish("K", v).expect("falsy parse"));
        }
    }

    #[test]
    fn parse_boolish_rejects_invalid_values() {
        assert!(parse_boolish("K", "maybe").is_err());
    }

    #[test]
    fn parse_conn_str_field_is_case_insensitive_and_trimmed() {
        let cs = " AccountName = acc ; AccountKey = key ; BlobEndpoint = http://127.0.0.1:10000/devstoreaccount1 ";
        assert_eq!(
            parse_conn_str_field(cs, "accountname").as_deref(),
            Some("acc")
        );
        assert_eq!(
            parse_conn_str_field(cs, "AccountKey").as_deref(),
            Some("key")
        );
        assert_eq!(
            parse_conn_str_field(cs, "BlobEndpoint").as_deref(),
            Some("http://127.0.0.1:10000/devstoreaccount1")
        );
    }
}
