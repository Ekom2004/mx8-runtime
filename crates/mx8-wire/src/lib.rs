#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use mx8_core::types as core;
use mx8_proto::v0 as wire;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConvertError {
    #[error("{field} must be non-empty")]
    EmptyField { field: &'static str },
    #[error("{field} is required")]
    MissingField { field: &'static str },
}

fn non_empty(field: &'static str, value: &str) -> Result<(), ConvertError> {
    if value.trim().is_empty() {
        return Err(ConvertError::EmptyField { field });
    }
    Ok(())
}

pub trait ToWire<T> {
    fn to_wire(&self) -> T;
}

pub trait ToCore<T> {
    fn to_core(&self) -> T;
}

pub trait TryToCore<T> {
    type Error;
    fn try_to_core(&self) -> Result<T, Self::Error>;
}

impl ToWire<wire::NodeCaps> for core::NodeCaps {
    fn to_wire(&self) -> wire::NodeCaps {
        wire::NodeCaps {
            max_fetch_concurrency: self.max_fetch_concurrency,
            max_decode_concurrency: self.max_decode_concurrency,
            max_inflight_bytes: self.max_inflight_bytes,
            max_ram_bytes: self.max_ram_bytes,
        }
    }
}

impl ToCore<core::NodeCaps> for wire::NodeCaps {
    fn to_core(&self) -> core::NodeCaps {
        core::NodeCaps {
            max_fetch_concurrency: self.max_fetch_concurrency,
            max_decode_concurrency: self.max_decode_concurrency,
            max_inflight_bytes: self.max_inflight_bytes,
            max_ram_bytes: self.max_ram_bytes,
        }
    }
}

impl ToWire<wire::NodeStats> for core::NodeStats {
    fn to_wire(&self) -> wire::NodeStats {
        wire::NodeStats {
            inflight_bytes: self.inflight_bytes,
            ram_high_water_bytes: self.ram_high_water_bytes,
            fetch_queue_depth: self.fetch_queue_depth,
            decode_queue_depth: self.decode_queue_depth,
            pack_queue_depth: self.pack_queue_depth,
            autotune_enabled: false,
            effective_prefetch_batches: 0,
            effective_max_queue_batches: 0,
            effective_want: 0,
            autotune_pressure_milli: 0,
            autotune_cooldown_ticks: 0,
            batch_payload_p95_over_p50_milli: 0,
            batch_jitter_slo_breaches_total: 0,
        }
    }
}

impl ToCore<core::NodeStats> for wire::NodeStats {
    fn to_core(&self) -> core::NodeStats {
        core::NodeStats {
            inflight_bytes: self.inflight_bytes,
            ram_high_water_bytes: self.ram_high_water_bytes,
            fetch_queue_depth: self.fetch_queue_depth,
            decode_queue_depth: self.decode_queue_depth,
            pack_queue_depth: self.pack_queue_depth,
        }
    }
}

impl ToWire<wire::WorkRange> for core::WorkRange {
    fn to_wire(&self) -> wire::WorkRange {
        wire::WorkRange {
            start_id: self.start_id,
            end_id: self.end_id,
            epoch: self.epoch,
            seed: self.seed,
        }
    }
}

impl ToCore<core::WorkRange> for wire::WorkRange {
    fn to_core(&self) -> core::WorkRange {
        core::WorkRange {
            start_id: self.start_id,
            end_id: self.end_id,
            epoch: self.epoch,
            seed: self.seed,
        }
    }
}

impl TryToCore<core::Lease> for wire::Lease {
    type Error = ConvertError;

    fn try_to_core(&self) -> Result<core::Lease, Self::Error> {
        non_empty("lease_id", &self.lease_id)?;
        non_empty("node_id", &self.node_id)?;

        let range = self
            .range
            .clone()
            .ok_or(ConvertError::MissingField { field: "range" })?;

        Ok(core::Lease {
            lease_id: core::LeaseId(self.lease_id.clone()),
            node_id: core::NodeId(self.node_id.clone()),
            range: range.to_core(),
            cursor: self.cursor,
            expires_unix_time_ms: self.expires_unix_time_ms,
        })
    }
}

impl ToWire<wire::Lease> for core::Lease {
    fn to_wire(&self) -> wire::Lease {
        wire::Lease {
            lease_id: self.lease_id.0.clone(),
            node_id: self.node_id.0.clone(),
            range: Some(self.range.to_wire()),
            cursor: self.cursor,
            expires_unix_time_ms: self.expires_unix_time_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_requires_range() {
        let lease = wire::Lease {
            lease_id: "l".to_string(),
            node_id: "n".to_string(),
            range: None,
            cursor: 0,
            expires_unix_time_ms: 0,
        };

        let err = lease.try_to_core().unwrap_err();
        assert_eq!(err, ConvertError::MissingField { field: "range" });
    }

    #[test]
    fn lease_roundtrip_core_wire() {
        let core_lease = core::Lease {
            lease_id: core::LeaseId("l".to_string()),
            node_id: core::NodeId("n".to_string()),
            range: core::WorkRange {
                start_id: 1,
                end_id: 2,
                epoch: 3,
                seed: 4,
            },
            cursor: 1,
            expires_unix_time_ms: 999,
        };

        let wire_lease = core_lease.to_wire();
        let decoded = wire_lease.try_to_core().unwrap();
        assert_eq!(decoded, core_lease);
    }
}
