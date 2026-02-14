use std::sync::Arc;

/// A batch is the unit of delivery to the consumer.
#[derive(Debug, Clone)]
pub struct Batch {
    pub sample_ids: Arc<[u64]>,
    /// Optional per-sample label IDs aligned with `sample_ids`.
    ///
    /// v0: this is populated when the manifest's `decode_hint` encodes
    /// `mx8:vision:imagefolder;label_id=<n>;...` for every sample in the batch.
    pub label_ids: Option<Arc<[u64]>>,
    /// Prefix-sum offsets into `payload` for each sample (length = sample_count + 1).
    ///
    /// Invariants:
    /// - offsets[0] == 0
    /// - offsets is non-decreasing
    /// - offsets.last() == payload.len()
    pub offsets: Arc<[u64]>,
    pub payload: Arc<[u8]>,
}

impl Batch {
    pub fn payload_len(&self) -> usize {
        self.payload.len()
    }

    pub fn sample_count(&self) -> usize {
        self.sample_ids.len()
    }

    pub fn offsets_len(&self) -> usize {
        self.offsets.len()
    }
}
