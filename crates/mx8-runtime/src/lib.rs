#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use anyhow::Result;

pub struct Runtime;

impl Runtime {
    pub fn new() -> Self {
        Self
    }

    pub fn run_once(&self) -> Result<()> {
        tracing::debug!("mx8 runtime placeholder");
        Ok(())
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}
