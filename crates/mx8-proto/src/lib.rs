#![forbid(unsafe_code)]

pub mod v0 {
    #![allow(clippy::expect_used, clippy::unwrap_used)]
    tonic::include_proto!("mx8.v0");
}
