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

