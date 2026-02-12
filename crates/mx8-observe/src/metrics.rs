use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug, Default)]
pub struct Counter(AtomicU64);

impl Counter {
    pub fn inc(&self) {
        self.inc_by(1);
    }

    pub fn inc_by(&self, value: u64) {
        self.0.fetch_add(value, Ordering::Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Default)]
pub struct Gauge(AtomicU64);

impl Gauge {
    pub fn set(&self, value: u64) {
        self.0.store(value, Ordering::Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Default)]
pub struct DurationAgg {
    count: AtomicU64,
    total_ns: AtomicU64,
    max_ns: AtomicU64,
}

impl DurationAgg {
    pub fn record(&self, dur: Duration) {
        let ns = dur.as_nanos().min(u64::MAX as u128) as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(ns, Ordering::Relaxed);

        let mut prev = self.max_ns.load(Ordering::Relaxed);
        while ns > prev {
            match self
                .max_ns
                .compare_exchange_weak(prev, ns, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(next) => prev = next,
            }
        }
    }

    pub fn snapshot(&self) -> DurationAggSnapshot {
        let count = self.count.load(Ordering::Relaxed);
        let total_ns = self.total_ns.load(Ordering::Relaxed);
        let max_ns = self.max_ns.load(Ordering::Relaxed);
        DurationAggSnapshot {
            count,
            total_ns,
            max_ns,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DurationAggSnapshot {
    pub count: u64,
    pub total_ns: u64,
    pub max_ns: u64,
}

impl DurationAggSnapshot {
    pub fn avg_ns(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.total_ns / self.count
        }
    }
}

pub struct ScopedTimer<'a> {
    start: Instant,
    agg: &'a DurationAgg,
}

impl<'a> ScopedTimer<'a> {
    pub fn new(agg: &'a DurationAgg) -> Self {
        Self {
            start: Instant::now(),
            agg,
        }
    }
}

impl Drop for ScopedTimer<'_> {
    fn drop(&mut self) {
        self.agg.record(self.start.elapsed());
    }
}
