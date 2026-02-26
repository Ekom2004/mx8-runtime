use anyhow::Result;

#[cfg(feature = "gcs")]
use google_cloud_storage::client::{Client, ClientConfig};

/// Build a GCS client from Application Default Credentials (ADC).
///
/// ADC resolution order:
///   1. `GOOGLE_APPLICATION_CREDENTIALS` env var → service account JSON path
///   2. gcloud user credentials  (`~/.config/gcloud/application_default_credentials.json`)
///   3. GCE / GKE workload identity metadata server
///
/// Override the GCS endpoint via `MX8_GCS_ENDPOINT_URL` (useful for local emulators
/// such as `fake-gcs-server`).  When set, the path-style URL
/// `http://localhost:4443/storage/v1` is typical.
#[cfg(feature = "gcs")]
pub async fn client_from_env() -> Result<Client> {
    let mut config = ClientConfig::default()
        .with_auth()
        .await
        .map_err(|e| anyhow::anyhow!("GCS ADC authentication failed: {e:?}"))?;

    if let Ok(url) = std::env::var("MX8_GCS_ENDPOINT_URL") {
        config.storage_endpoint = url;
    }

    Ok(Client::new(config))
}
