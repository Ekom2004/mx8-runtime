use std::path::{Path, PathBuf};
use std::time::Duration;

use mx8_core::types::ManifestHash;

use crate::{
    unix_time_ms, validate_intent_key, validate_manifest_hash, write_atomic, ManifestStore,
    ManifestStoreError,
};

#[derive(Debug, Clone)]
pub struct FsManifestStore {
    root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct LockOwner {
    pub node_id: Option<String>,
}

#[derive(Debug)]
pub struct FsIndexLockGuard {
    path: PathBuf,
}

impl Drop for FsIndexLockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl FsManifestStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn intent_key_for_base(base: &str) -> String {
        let normalized = normalize_intent_base(base);
        let hash = fnv1a64(normalized.as_bytes());
        format!("fnv64_{hash:016x}")
    }

    fn by_hash_path(&self, hash: &ManifestHash) -> Result<PathBuf, ManifestStoreError> {
        if !validate_manifest_hash(hash) {
            return Err(ManifestStoreError::InvalidManifestHash);
        }
        Ok(self.root.join("by-hash").join(hash.0.as_str()))
    }

    fn intent_current_path(&self, intent_key: &str) -> Result<PathBuf, ManifestStoreError> {
        if !validate_intent_key(intent_key) {
            return Err(ManifestStoreError::InvalidIntentKey);
        }
        Ok(self.root.join("intent").join(intent_key).join("current"))
    }

    fn lock_path(&self, intent_key: &str) -> Result<PathBuf, ManifestStoreError> {
        if !validate_intent_key(intent_key) {
            return Err(ManifestStoreError::InvalidIntentKey);
        }
        Ok(self.root.join("locks").join(format!("{intent_key}.lock")))
    }

    fn read_to_string(path: &Path) -> Result<String, ManifestStoreError> {
        let bytes = std::fs::read(path)?;
        let s = String::from_utf8_lossy(&bytes);
        Ok(s.trim().to_string())
    }

    pub fn try_acquire_index_lock(
        &self,
        intent_key: &str,
        stale_after: Duration,
        owner: LockOwner,
    ) -> Result<Option<FsIndexLockGuard>, ManifestStoreError> {
        let path = self.lock_path(intent_key)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let now_ms = unix_time_ms();
        let content = lock_file_content(now_ms, &owner);

        match std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
        {
            Ok(mut f) => {
                use std::io::Write;
                f.write_all(content.as_bytes())?;
                f.sync_all()?;
                Ok(Some(FsIndexLockGuard { path }))
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                if stale_after.is_zero() {
                    return Ok(None);
                }

                if is_lock_stale(&path, stale_after, now_ms)? {
                    let _ = std::fs::remove_file(&path);
                    match std::fs::OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .open(&path)
                    {
                        Ok(mut f) => {
                            use std::io::Write;
                            f.write_all(content.as_bytes())?;
                            f.sync_all()?;
                            Ok(Some(FsIndexLockGuard { path }))
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(None),
                        Err(err) => Err(ManifestStoreError::Io(err)),
                    }
                } else {
                    Ok(None)
                }
            }
            Err(err) => Err(ManifestStoreError::Io(err)),
        }
    }
}

impl ManifestStore for FsManifestStore {
    fn put_manifest_bytes(
        &self,
        hash: &ManifestHash,
        bytes: &[u8],
    ) -> Result<(), ManifestStoreError> {
        let path = self.by_hash_path(hash)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        match std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
        {
            Ok(mut f) => {
                use std::io::Write;
                f.write_all(bytes)?;
                f.sync_all()?;
                Ok(())
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                let existing = std::fs::read(&path)?;
                if existing == bytes {
                    Ok(())
                } else {
                    Err(ManifestStoreError::HashCollision {
                        hash: hash.0.clone(),
                    })
                }
            }
            Err(err) => Err(ManifestStoreError::Io(err)),
        }
    }

    fn get_manifest_bytes(&self, hash: &ManifestHash) -> Result<Vec<u8>, ManifestStoreError> {
        let path = self.by_hash_path(hash)?;
        match std::fs::read(&path) {
            Ok(bytes) => Ok(bytes),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Err(ManifestStoreError::NotFound(hash.0.clone()))
            }
            Err(err) => Err(ManifestStoreError::Io(err)),
        }
    }

    fn get_current_snapshot(
        &self,
        intent_key: &str,
    ) -> Result<Option<ManifestHash>, ManifestStoreError> {
        let path = self.intent_current_path(intent_key)?;
        match Self::read_to_string(&path) {
            Ok(s) if s.is_empty() => Ok(None),
            Ok(s) => Ok(Some(ManifestHash(s))),
            Err(ManifestStoreError::Io(err)) if err.kind() == std::io::ErrorKind::NotFound => {
                Ok(None)
            }
            Err(err) => Err(err),
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

        let path = self.intent_current_path(intent_key)?;
        write_atomic(&path, hash.0.as_bytes())?;
        Ok(())
    }
}

fn normalize_intent_base(base: &str) -> String {
    let trimmed = base.trim();
    trimmed.trim_end_matches('/').to_string()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn lock_file_content(unix_time_ms: u64, owner: &LockOwner) -> String {
    let node_id = owner.node_id.as_deref().unwrap_or("");
    format!(
        "unix_time_ms={unix_time_ms}\npid={}\nnode_id={node_id}\n",
        std::process::id()
    )
}

fn is_lock_stale(
    path: &Path,
    stale_after: Duration,
    now_ms: u64,
) -> Result<bool, ManifestStoreError> {
    let content = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(ManifestStoreError::Io(err)),
    };

    let s = String::from_utf8_lossy(&content);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn temp_root(test_name: &str) -> anyhow::Result<PathBuf> {
        let mut root = std::env::temp_dir();
        let suffix = format!(
            "mx8-manifest-store-{}-{}-{}",
            test_name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );
        root.push(suffix);
        std::fs::create_dir_all(&root)?;
        Ok(root)
    }

    #[test]
    fn manifest_is_immutable_by_hash() -> anyhow::Result<()> {
        let root = temp_root("immutable")?;
        let store = FsManifestStore::new(root);

        let hash = ManifestHash("abc123".to_string());
        store.put_manifest_bytes(&hash, b"hello")?;
        store.put_manifest_bytes(&hash, b"hello")?;

        let err = store.put_manifest_bytes(&hash, b"goodbye").unwrap_err();
        match err {
            ManifestStoreError::HashCollision { hash: h } => assert_eq!(h, "abc123"),
            other => panic!("expected HashCollision, got {other:?}"),
        }

        Ok(())
    }

    #[test]
    fn intent_pointer_updates() -> anyhow::Result<()> {
        let root = temp_root("intent-pointer")?;
        let store = FsManifestStore::new(root);

        let intent_key = "datasetA";
        assert_eq!(store.get_current_snapshot(intent_key)?, None);

        let h1 = ManifestHash("h1".to_string());
        store.set_current_snapshot(intent_key, &h1)?;
        assert_eq!(store.get_current_snapshot(intent_key)?, Some(h1.clone()));

        let h2 = ManifestHash("h2".to_string());
        store.set_current_snapshot(intent_key, &h2)?;
        assert_eq!(store.get_current_snapshot(intent_key)?, Some(h2));

        Ok(())
    }

    #[test]
    fn index_lock_allows_single_winner() -> anyhow::Result<()> {
        let root = temp_root("lock-single-winner")?;
        let store = Arc::new(FsManifestStore::new(root));

        let intent_key = "intentA";
        let threads = 12;
        let barrier = Arc::new(Barrier::new(threads));
        let winners = Arc::new(AtomicU64::new(0));

        let mut handles = Vec::new();
        for _ in 0..threads {
            let store = store.clone();
            let barrier = barrier.clone();
            let winners = winners.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                let guard = store
                    .try_acquire_index_lock(
                        intent_key,
                        Duration::from_secs(60),
                        LockOwner { node_id: None },
                    )
                    .expect("lock acquisition should not error");
                if guard.is_some() {
                    winners.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(50));
                }
            }));
        }

        for h in handles {
            h.join().expect("thread join");
        }

        assert_eq!(winners.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn stale_lock_is_reaped() -> anyhow::Result<()> {
        let root = temp_root("lock-stale-reap")?;
        let store = FsManifestStore::new(root.clone());

        let intent_key = "intentB";
        let lock_path = store.lock_path(intent_key)?;
        if let Some(parent) = lock_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            &lock_path,
            lock_file_content(0, &LockOwner { node_id: None }),
        )?;

        let guard = store.try_acquire_index_lock(
            intent_key,
            Duration::from_millis(1),
            LockOwner { node_id: None },
        )?;
        assert!(guard.is_some(), "expected to reap stale lock");

        Ok(())
    }

    #[test]
    fn intent_key_normalizes_trailing_slash() {
        let a = FsManifestStore::intent_key_for_base("s3://bucket/prefix");
        let b = FsManifestStore::intent_key_for_base("s3://bucket/prefix/");
        assert_eq!(a, b);
    }
}
