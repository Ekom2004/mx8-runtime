use anyhow::Result;

#[cfg(feature = "s3")]
use aws_sdk_s3::config::Builder as S3ConfigBuilder;

#[cfg(feature = "s3")]
fn parse_env_bool(key: &str) -> Result<Option<bool>> {
    match std::env::var(key) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            let b = match s.as_str() {
                "1" | "true" | "yes" | "y" | "on" => true,
                "0" | "false" | "no" | "n" | "off" => false,
                _ => anyhow::bail!(
                    "invalid boolean env var {}={:?} (expected true/false/1/0)",
                    key,
                    v
                ),
            };
            Ok(Some(b))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(anyhow::Error::new(e)),
    }
}

/// Build an S3 client from the ambient environment.
///
/// v0 behavior:
/// - Default: standard AWS resolution (region/creds from env/config/role).
/// - Optional: override endpoint via `MX8_S3_ENDPOINT_URL` (for MinIO or other S3-compatible stores).
/// - Optional: `MX8_S3_FORCE_PATH_STYLE=1` to force path-style addressing (recommended for MinIO).
#[cfg(feature = "s3")]
pub async fn client_from_env() -> Result<aws_sdk_s3::Client> {
    let cfg = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;

    let endpoint_url: Option<String> = std::env::var("MX8_S3_ENDPOINT_URL").ok();
    let force_path_style = match parse_env_bool("MX8_S3_FORCE_PATH_STYLE")? {
        Some(v) => v,
        None => endpoint_url.is_some(),
    };

    let mut b: S3ConfigBuilder = aws_sdk_s3::config::Builder::from(&cfg);

    if let Some(url) = endpoint_url {
        b = b.endpoint_url(url);
    }
    if force_path_style {
        b = b.force_path_style(true);
    }

    Ok(aws_sdk_s3::Client::from_conf(b.build()))
}
