use std::future::Future;
use std::time::Duration;

use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::primitives::{AggregatedBytes, ByteStream};

use crate::{
    unix_time_ms, validate_intent_key, validate_manifest_hash, IndexLockGuard, LockOwner,
    ManifestStore, ManifestStoreError,
};
use mx8_core::types::ManifestHash;

#[derive(Debug, Clone)]
pub struct S3ManifestStore {
    client: aws_sdk_s3::Client,
    bucket: String,
    prefix: String,
}

#[derive(Debug)]
struct S3IndexLockGuard {
    client: aws_sdk_s3::Client,
    bucket: String,
    key: String,
}

impl IndexLockGuard for S3IndexLockGuard {}

impl Drop for S3IndexLockGuard {
    fn drop(&mut self) {
        let client = self.client.clone();
        let bucket = self.bucket.clone();
        let key = self.key.clone();
        let _ = block_on(async move {
            let _ = client.delete_object().bucket(bucket).key(key).send().await;
            Ok::<(), ManifestStoreError>(())
        });
    }
}

impl S3ManifestStore {
    /// Create an S3-backed manifest store from a `s3://bucket/prefix` URL with the leading
    /// scheme stripped (i.e. pass `bucket/prefix`).
    pub fn from_env_url(rest: &str) -> Result<Self, ManifestStoreError> {
        let (bucket, prefix) = parse_bucket_prefix(rest)?;

        let client = block_on(client_from_env())??;

        // Best-effort bucket creation (ignore "already exists/owned" errors).
        let _ = block_on({
            let c = client.clone();
            let b = bucket.clone();
            async move {
                let _ = c.create_bucket().bucket(b).send().await;
                Ok::<(), ManifestStoreError>(())
            }
        });

        Ok(Self {
            client,
            bucket,
            prefix,
        })
    }

    fn key_by_hash(&self, hash: &ManifestHash) -> Result<String, ManifestStoreError> {
        if !validate_manifest_hash(hash) {
            return Err(ManifestStoreError::InvalidManifestHash);
        }
        Ok(self.join_key(&["by-hash", hash.0.as_str()]))
    }

    fn key_intent_current(&self, intent_key: &str) -> Result<String, ManifestStoreError> {
        if !validate_intent_key(intent_key) {
            return Err(ManifestStoreError::InvalidIntentKey);
        }
        Ok(self.join_key(&["intent", intent_key, "current"]))
    }

    fn key_lock(&self, intent_key: &str) -> Result<String, ManifestStoreError> {
        if !validate_intent_key(intent_key) {
            return Err(ManifestStoreError::InvalidIntentKey);
        }
        Ok(self.join_key(&["locks", &format!("{intent_key}.lock")]))
    }

    fn join_key(&self, parts: &[&str]) -> String {
        let mut out = String::new();
        if !self.prefix.is_empty() {
            out.push_str(self.prefix.trim_matches('/'));
        }
        for p in parts {
            if !out.is_empty() {
                out.push('/');
            }
            out.push_str(p.trim_matches('/'));
        }
        out
    }

    async fn read_object_bytes(&self, key: &str) -> Result<Vec<u8>, ManifestStoreError> {
        let out = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(map_get_err)?;
        let bytes: AggregatedBytes = out.body.collect().await.map_err(|e| {
            ManifestStoreError::Runtime(format!("get_object body collect failed: {e:?}"))
        })?;
        Ok(bytes.into_bytes().to_vec())
    }

    async fn lock_is_stale(
        &self,
        key: &str,
        stale_after: Duration,
        now_ms: u64,
    ) -> Result<bool, ManifestStoreError> {
        let bytes = match self.read_object_bytes(key).await {
            Ok(b) => b,
            Err(ManifestStoreError::NotFound(_)) => return Ok(false),
            Err(e) => return Err(e),
        };
        let s = String::from_utf8_lossy(&bytes);
        let mut lock_unix_ms: Option<u64> = None;
        for line in s.lines() {
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            if k.trim() == "unix_time_ms" {
                if let Ok(parsed) = v.trim().parse::<u64>() {
                    lock_unix_ms = Some(parsed);
                    break;
                }
            }
        }
        let Some(lock_ms) = lock_unix_ms else {
            return Ok(false);
        };
        let age_ms = now_ms.saturating_sub(lock_ms);
        Ok(age_ms >= stale_after.as_millis().min(u64::MAX as u128) as u64)
    }
}

impl ManifestStore for S3ManifestStore {
    fn try_acquire_index_lock(
        &self,
        intent_key: &str,
        stale_after: Duration,
        owner: LockOwner,
    ) -> Result<Option<Box<dyn IndexLockGuard>>, ManifestStoreError> {
        let key = self.key_lock(intent_key)?;
        let now_ms = unix_time_ms();
        let node_id = owner.node_id.as_deref().unwrap_or("");
        let content = format!(
            "unix_time_ms={now_ms}\npid={}\nnode_id={node_id}\n",
            std::process::id()
        );

        let b = self.bucket.clone();
        let k = key.clone();
        let put = block_on({
            let c = self.client.clone();
            let body = content.clone().into_bytes();
            async move {
                c.put_object()
                    .bucket(b)
                    .key(k)
                    .if_none_match("*")
                    .body(ByteStream::from(body))
                    .send()
                    .await
            }
        })?;

        match put {
            Ok(_) => {
                let guard = S3IndexLockGuard {
                    client: self.client.clone(),
                    bucket: self.bucket.clone(),
                    key,
                };
                Ok(Some(Box::new(guard)))
            }
            Err(err) => {
                if is_put_precondition_failed(&err) {
                    if stale_after.is_zero() {
                        return Ok(None);
                    }

                    if block_on(self.lock_is_stale(&key, stale_after, now_ms))?? {
                        let _ = block_on({
                            let c = self.client.clone();
                            let b = self.bucket.clone();
                            let k = key.clone();
                            async move {
                                let _ = c.delete_object().bucket(b).key(k).send().await;
                                Ok::<(), ManifestStoreError>(())
                            }
                        });

                        // Retry once after stale lock reaping.
                        let b = self.bucket.clone();
                        let k = key.clone();
                        let put2 = block_on({
                            let c = self.client.clone();
                            let body = content.into_bytes();
                            async move {
                                c.put_object()
                                    .bucket(b)
                                    .key(k)
                                    .if_none_match("*")
                                    .body(ByteStream::from(body))
                                    .send()
                                    .await
                            }
                        })?;

                        match put2 {
                            Ok(_) => {
                                let guard = S3IndexLockGuard {
                                    client: self.client.clone(),
                                    bucket: self.bucket.clone(),
                                    key,
                                };
                                return Ok(Some(Box::new(guard)));
                            }
                            Err(err2) if is_put_precondition_failed(&err2) => return Ok(None),
                            Err(err2) => return Err(map_put_err(err2)),
                        }
                    }
                    return Ok(None);
                }

                Err(map_put_err(err))
            }
        }
    }

    fn put_manifest_bytes(
        &self,
        hash: &ManifestHash,
        bytes: &[u8],
    ) -> Result<(), ManifestStoreError> {
        let key = self.key_by_hash(hash)?;
        let b = self.bucket.clone();
        let k = key.clone();
        let body = bytes.to_vec();

        let put = block_on({
            let c = self.client.clone();
            async move {
                c.put_object()
                    .bucket(b)
                    .key(k)
                    .if_none_match("*")
                    .body(ByteStream::from(body))
                    .send()
                    .await
            }
        })?;

        match put {
            Ok(_) => Ok(()),
            Err(err) if is_put_precondition_failed(&err) => {
                let existing = self.get_manifest_bytes(hash)?;
                if existing == bytes {
                    Ok(())
                } else {
                    Err(ManifestStoreError::HashCollision {
                        hash: hash.0.clone(),
                    })
                }
            }
            Err(err) => Err(map_put_err(err)),
        }
    }

    fn put_manifest_file(
        &self,
        hash: &ManifestHash,
        path: &std::path::Path,
    ) -> Result<(), ManifestStoreError> {
        let key = self.key_by_hash(hash)?;
        let b = self.bucket.clone();
        let k = key.clone();
        let path_buf = path.to_path_buf();

        let put = block_on({
            let c = self.client.clone();
            async move {
                let body = match ByteStream::from_path(path_buf).await {
                    Ok(body) => body,
                    Err(err) => {
                        return Err(ManifestStoreError::Runtime(format!(
                            "create bytestream from file failed: {err:?}"
                        )));
                    }
                };
                let out = c
                    .put_object()
                    .bucket(b)
                    .key(k)
                    .if_none_match("*")
                    .body(body)
                    .send()
                    .await;
                Ok::<_, ManifestStoreError>(out)
            }
        })??;

        match put {
            Ok(_) => Ok(()),
            Err(err) if is_put_precondition_failed(&err) => {
                let existing = self.get_manifest_bytes(hash)?;
                let existing_hash = crate::sha256_hex(&existing);
                let incoming_hash = hash_file_sha256(path)?;
                if existing_hash == incoming_hash {
                    Ok(())
                } else {
                    Err(ManifestStoreError::HashCollision {
                        hash: hash.0.clone(),
                    })
                }
            }
            Err(err) => Err(map_put_err(err)),
        }
    }

    fn get_manifest_bytes(&self, hash: &ManifestHash) -> Result<Vec<u8>, ManifestStoreError> {
        let key = self.key_by_hash(hash)?;
        match block_on(self.read_object_bytes(&key))? {
            Ok(bytes) => Ok(bytes),
            Err(ManifestStoreError::NotFound(_)) => {
                Err(ManifestStoreError::NotFound(hash.0.clone()))
            }
            Err(e) => Err(e),
        }
    }

    fn get_manifest_len(&self, hash: &ManifestHash) -> Result<u64, ManifestStoreError> {
        let key = self.key_by_hash(hash)?;
        let out = block_on({
            let c = self.client.clone();
            let bucket = self.bucket.clone();
            let key = key.clone();
            async move { c.head_object().bucket(bucket).key(key).send().await }
        })?;
        match out {
            Ok(head) => Ok(head.content_length().unwrap_or(0) as u64),
            Err(err) => match map_head_err(err) {
                ManifestStoreError::NotFound(_) => {
                    Err(ManifestStoreError::NotFound(hash.0.clone()))
                }
                other => Err(other),
            },
        }
    }

    fn get_manifest_range(
        &self,
        hash: &ManifestHash,
        offset: u64,
        len: usize,
    ) -> Result<Vec<u8>, ManifestStoreError> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let key = self.key_by_hash(hash)?;
        let end = offset.saturating_add(len as u64).saturating_sub(1);
        let range = format!("bytes={offset}-{end}");
        let out = block_on({
            let c = self.client.clone();
            let bucket = self.bucket.clone();
            let key = key.clone();
            let range = range.clone();
            async move {
                c.get_object()
                    .bucket(bucket)
                    .key(key)
                    .range(range)
                    .send()
                    .await
            }
        })?;
        match out {
            Ok(resp) => {
                let bytes: AggregatedBytes = block_on(async move {
                    resp.body.collect().await.map_err(|e| {
                        ManifestStoreError::Runtime(format!(
                            "get_object range body collect failed: {e:?}"
                        ))
                    })
                })??;
                Ok(bytes.into_bytes().to_vec())
            }
            Err(err) => match map_get_err(err) {
                ManifestStoreError::NotFound(_) => {
                    Err(ManifestStoreError::NotFound(hash.0.clone()))
                }
                other => Err(other),
            },
        }
    }

    fn get_current_snapshot(
        &self,
        intent_key: &str,
    ) -> Result<Option<ManifestHash>, ManifestStoreError> {
        let key = self.key_intent_current(intent_key)?;
        match block_on(self.read_object_bytes(&key))? {
            Ok(bytes) => {
                let s = String::from_utf8_lossy(&bytes).trim().to_string();
                if s.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(ManifestHash(s)))
                }
            }
            Err(ManifestStoreError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn set_current_snapshot(
        &self,
        intent_key: &str,
        hash: &ManifestHash,
    ) -> Result<(), ManifestStoreError> {
        if !validate_manifest_hash(hash) {
            return Err(ManifestStoreError::InvalidManifestHash);
        }
        let key = self.key_intent_current(intent_key)?;
        let b = self.bucket.clone();
        let k = key.clone();
        let body = hash.0.as_bytes().to_vec();

        block_on({
            let c = self.client.clone();
            async move {
                c.put_object()
                    .bucket(b)
                    .key(k)
                    .body(ByteStream::from(body))
                    .send()
                    .await
                    .map_err(map_put_err)?;
                Ok::<(), ManifestStoreError>(())
            }
        })??;
        Ok(())
    }
}

fn parse_bucket_prefix(rest: &str) -> Result<(String, String), ManifestStoreError> {
    let s = rest.trim().trim_matches('/');
    let mut it = s.splitn(2, '/');
    let bucket = it.next().unwrap_or("").trim();
    if bucket.is_empty() {
        return Err(ManifestStoreError::UnsupportedRoot(format!(
            "invalid s3 manifest_store root: s3://{rest}"
        )));
    }
    let prefix = it.next().unwrap_or("").trim_matches('/').to_string();
    Ok((bucket.to_string(), prefix))
}

async fn client_from_env() -> Result<aws_sdk_s3::Client, ManifestStoreError> {
    let cfg = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;

    let endpoint_url: Option<String> = std::env::var("MX8_S3_ENDPOINT_URL").ok();
    let force_path_style = match parse_env_bool("MX8_S3_FORCE_PATH_STYLE")? {
        Some(v) => v,
        None => endpoint_url.is_some(),
    };

    let mut b = aws_sdk_s3::config::Builder::from(&cfg);
    if let Some(url) = endpoint_url {
        b = b.endpoint_url(url);
    }
    if force_path_style {
        b = b.force_path_style(true);
    }

    Ok(aws_sdk_s3::Client::from_conf(b.build()))
}

fn parse_env_bool(key: &str) -> Result<Option<bool>, ManifestStoreError> {
    match std::env::var(key) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            let b = match s.as_str() {
                "1" | "true" | "yes" | "y" | "on" => true,
                "0" | "false" | "no" | "n" | "off" => false,
                _ => {
                    return Err(ManifestStoreError::Runtime(format!(
                        "invalid boolean env var {}={:?} (expected true/false/1/0)",
                        key, v
                    )))
                }
            };
            Ok(Some(b))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(ManifestStoreError::Runtime(format!(
            "read env var {key} failed: {e}"
        ))),
    }
}

fn block_on<Fut>(fut: Fut) -> Result<Fut::Output, ManifestStoreError>
where
    Fut: Future,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => Ok(tokio::task::block_in_place(|| handle.block_on(fut))),
        Err(_) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    ManifestStoreError::Runtime(format!("tokio runtime init failed: {e}"))
                })?;
            Ok(rt.block_on(fut))
        }
    }
}

fn is_put_precondition_failed(
    err: &aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::put_object::PutObjectError>,
) -> bool {
    match err {
        aws_sdk_s3::error::SdkError::ServiceError(se) => {
            se.err().code() == Some("PreconditionFailed")
        }
        _ => false,
    }
}

fn map_put_err(
    err: aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::put_object::PutObjectError>,
) -> ManifestStoreError {
    ManifestStoreError::Runtime(format!("s3 put_object failed: {err:?}"))
}

fn map_get_err(
    err: aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::get_object::GetObjectError>,
) -> ManifestStoreError {
    match err {
        aws_sdk_s3::error::SdkError::ServiceError(ref se) => {
            if se.err().is_no_such_key() {
                ManifestStoreError::NotFound("no such key".to_string())
            } else {
                ManifestStoreError::Runtime(format!("s3 get_object service error: {err:?}"))
            }
        }
        other => ManifestStoreError::Runtime(format!("s3 get_object failed: {other:?}")),
    }
}

fn map_head_err(
    err: aws_sdk_s3::error::SdkError<aws_sdk_s3::operation::head_object::HeadObjectError>,
) -> ManifestStoreError {
    match err {
        aws_sdk_s3::error::SdkError::ServiceError(ref se) => {
            if se.err().is_not_found() {
                ManifestStoreError::NotFound("no such key".to_string())
            } else {
                ManifestStoreError::Runtime(format!("s3 head_object service error: {err:?}"))
            }
        }
        other => ManifestStoreError::Runtime(format!("s3 head_object failed: {other:?}")),
    }
}

fn hash_file_sha256(path: &std::path::Path) -> Result<String, ManifestStoreError> {
    use std::io::Read as _;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = crate::sha256_streaming_new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(crate::sha256_to_lower_hex(&digest))
}
