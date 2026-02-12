use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkRange {
    pub start_id: u64,
    pub end_id: u64, // half-open [start_id, end_id)
    pub epoch: u32,
    pub seed: u64,
}

impl WorkRange {
    pub fn len(&self) -> u64 {
        self.end_id.saturating_sub(self.start_id)
    }

    pub fn is_empty(&self) -> bool {
        self.start_id >= self.end_id
    }

    pub fn contains(&self, sample_id: u64) -> bool {
        self.start_id <= sample_id && sample_id < self.end_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    pub lease_id: String,
    pub node_id: String,
    pub range: WorkRange,
    pub cursor: u64,
}
