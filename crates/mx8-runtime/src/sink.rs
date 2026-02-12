use anyhow::Result;

use crate::types::Batch;

/// Delivery interface for `mx8-runtime`.
///
/// This is intentionally synchronous for the first vertical slice. A slow sink must exert
/// backpressure (i.e., block upstream) so the runtime stays RAM-bounded.
pub trait Sink: Send + Sync + 'static {
    fn deliver(&self, batch: Batch) -> Result<()>;
}
