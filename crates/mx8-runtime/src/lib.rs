#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]
#![allow(unexpected_cfgs)]

#[cfg(feature = "azure")]
pub mod azure;
#[cfg(feature = "gcs")]
pub mod gcs;
pub mod pipeline;
#[cfg(feature = "s3")]
pub mod s3;
pub mod sink;
pub mod types;
