//! Per-bucket captured-graph pool.
//!
//! Graphs are captured at engine init (not lazily during warmup).
//! Each bucket stores (graph handle, layout hash, fingerprint). On
//! replay, the runtime asserts the current bucket's `MetadataLayout`
//! hash still matches the captured one — drift = typed err.

use std::collections::BTreeMap;

use rvllm_core::{GraphError, MetaLayoutHash, Result, RvllmError};
use rvllm_metadata::MetadataLayout;

/// The sha256 of a captured graph's structural signature — kernel
/// names and their arg-scalar values, in the order they were recorded.
/// Stored so a debug-build replay can re-walk the graph and verify.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct GraphFingerprint(pub [u8; 32]);

/// One captured graph.
#[derive(Debug)]
pub struct CapturedGraph {
    pub bucket: u32,
    pub max_blocks: u32,
    pub layout_hash: MetaLayoutHash,
    pub fingerprint: GraphFingerprint,
    // Real runtime keeps a cuGraphExec handle here behind
    // `feature = "cuda"`.
}

/// Pool of graphs keyed by `(bucket, max_blocks)`.
#[derive(Default, Debug)]
pub struct GraphPool {
    graphs: BTreeMap<(u32, u32), CapturedGraph>,
}

impl GraphPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, g: CapturedGraph) {
        self.graphs.insert((g.bucket, g.max_blocks), g);
    }

    pub fn get(&self, bucket: u32, max_blocks: u32) -> Option<&CapturedGraph> {
        self.graphs.get(&(bucket, max_blocks))
    }

    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Before replay, verify the current layout for `(bucket, max_blocks)`
    /// still matches the hash baked into the captured graph. Drift =
    /// typed error (graph must be re-captured or config is wrong).
    pub fn check_before_replay(
        &self,
        bucket: u32,
        max_blocks: u32,
        current: &MetadataLayout,
    ) -> Result<&CapturedGraph> {
        let g = self
            .get(bucket, max_blocks)
            .ok_or_else(|| RvllmError::graph(GraphError::BucketMissing { padded_batch: bucket }, bucket))?;
        let h = current.hash();
        if h != g.layout_hash {
            return Err(RvllmError::graph(
                GraphError::CaptureMetadataMismatch {
                    captured: g.layout_hash,
                    replay: h,
                },
                bucket,
            ));
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_graph(bucket: u32, max_blocks: u32) -> CapturedGraph {
        let layout = MetadataLayout::compute(bucket, max_blocks);
        CapturedGraph {
            bucket,
            max_blocks,
            layout_hash: layout.hash(),
            fingerprint: GraphFingerprint([0u8; 32]),
        }
    }

    #[test]
    fn matching_layout_replays() {
        let mut pool = GraphPool::new();
        pool.insert(fake_graph(128, 129));
        let layout = MetadataLayout::compute(128, 129);
        assert!(pool.check_before_replay(128, 129, &layout).is_ok());
    }

    #[test]
    fn drift_returns_typed_err() {
        let mut pool = GraphPool::new();
        pool.insert(fake_graph(128, 129));
        // Caller passes a different max_blocks → different hash.
        let wrong = MetadataLayout::compute(128, 257);
        let err = pool.check_before_replay(128, 129, &wrong).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("CaptureMetadataMismatch"));
    }

    #[test]
    fn missing_bucket_is_typed_err() {
        let pool = GraphPool::new();
        let layout = MetadataLayout::compute(1, 8);
        let err = pool.check_before_replay(1, 8, &layout).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("BucketMissing"));
        assert!(s.contains("padded_batch: 1"));
    }
}
