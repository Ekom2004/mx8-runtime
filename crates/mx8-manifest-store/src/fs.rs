use std::path::{Path, PathBuf};

use mx8_core::types::ManifestHash;

use crate::{
    validate_intent_key, validate_manifest_hash, write_atomic, ManifestStore, ManifestStoreError,
};

#[derive(Debug, Clone)]
pub struct FsManifestStore {
    root: PathBuf,
}

impl FsManifestStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
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

    fn read_to_string(path: &Path) -> Result<String, ManifestStoreError> {
        let bytes = std::fs::read(path)?;
        let s = String::from_utf8_lossy(&bytes);
        Ok(s.trim().to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
