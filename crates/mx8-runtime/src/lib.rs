#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

pub mod pipeline;
pub mod sink;
pub mod types;
