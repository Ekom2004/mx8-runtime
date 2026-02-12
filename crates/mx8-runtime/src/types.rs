use std::sync::Arc;

/// A batch is the unit of delivery to the consumer.
#[derive(Debug, Clone)]
pub struct Batch {
    pub sample_ids: Arc<[u64]>,
    pub payload: Arc<[u8]>,
}

impl Batch {
    pub fn payload_len(&self) -> usize {
        self.payload.len()
    }

    pub fn sample_count(&self) -> usize {
        self.sample_ids.len()
    }
}
