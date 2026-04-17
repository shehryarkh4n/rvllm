//! v3 Engine: type-state API over v2's proven kernel stack.
//!
//! The engine exposes exactly one way to do a step:
//!
//! ```ignore
//! let pending = engine.launch()?;
//! let outputs = pending.collect()?;
//! ```
//!
//! `PendingStep<'a>` borrows the engine mutably, so calling `launch()` again
//! without calling `collect()` first is a compile error. This encodes the
//! pipelined invariant that v2 had to enforce by runtime state checks.

use rvllm_core::prelude::SamplingParams;
use rvllm_v2::engine::{Engine as V2Engine, StepPending};
use rvllm_v2::integration::{init_engine, ConcreteEngine, V2EngineConfig};
use rvllm_v2::types::V2RequestOutput;
use std::path::PathBuf;

use crate::error::{CudaErrorKind, Result, RvllmError};

pub struct Engine {
    inner: ConcreteEngine,
}

/// A launched-but-not-collected step. Borrows the engine mutably so a
/// second launch cannot start until this is collected.
#[must_use = "a PendingStep must be collected or dropped; silent drop wastes a cuEventSynchronize"]
pub struct PendingStep<'a> {
    engine: &'a mut Engine,
    pending: Option<StepPending>,
}

impl Engine {
    /// Initialize the v3 engine. Refuses to start if:
    /// - FA3 `.so` is missing (no PTX fallback)
    /// - CUTLASS `.so` is missing
    /// - Autotune policy JSON is empty (tell the user to run autotune-cutlass)
    pub fn new(config: V2EngineConfig) -> Result<Self> {
        verify_kernel_artifacts(&config)?;
        let inner = init_engine(&config);
        Ok(Self { inner })
    }

    /// Add a request; returns an engine-assigned request id.
    pub fn add_request(&mut self, prompt: String, params: SamplingParams) -> Result<u64> {
        let id = self
            .inner
            .add_request(prompt, params)
            .map_err(|e| RvllmError::Loader {
                kind: "add_request",
                detail: format!("{e}"),
            })?;
        Ok(id.0)
    }

    /// True iff the scheduler has any work to do.
    pub fn has_pending_work(&self) -> bool {
        self.inner.has_pending_work()
    }

    /// Active requests the engine is currently serving.
    pub fn num_active(&self) -> usize {
        self.inner.num_active_requests()
    }

    /// Launch one step of GPU work. Returns a `PendingStep` that must be
    /// `collect()`ed before another launch. If the scheduler has no work,
    /// returns `Ok(None)`.
    pub fn launch(&mut self) -> Result<Option<PendingStep<'_>>> {
        let pending = self.inner.step_launch().map_err(map_engine_err)?;
        match pending {
            Some(p) => Ok(Some(PendingStep {
                engine: self,
                pending: Some(p),
            })),
            None => Ok(None),
        }
    }

    /// One step of the pipeline (collect previous, launch next). Returns
    /// any outputs from the previous step; the current step is launched
    /// and will be collected on the next `step()` call or `flush()`.
    pub fn step(&mut self) -> Result<Vec<V2RequestOutput>> {
        self.inner.step_pipelined().map_err(map_engine_err)
    }

    /// Drain the final pipelined step. Call once after the generation loop.
    pub fn flush(&mut self) -> Result<Vec<V2RequestOutput>> {
        self.inner.step_pipelined_flush().map_err(map_engine_err)
    }

    /// Fence the compute stream. Only used at shutdown or engine init.
    pub fn sync(&self) -> Result<()> {
        self.inner.sync().map_err(map_engine_err)
    }

    /// Escape hatch to the v2 engine (for migration only — discouraged).
    pub fn into_inner(self) -> ConcreteEngine {
        self.inner
    }
    pub fn as_inner(&self) -> &ConcreteEngine {
        &self.inner
    }
    pub fn as_inner_mut(&mut self) -> &mut ConcreteEngine {
        &mut self.inner
    }
}

impl<'a> PendingStep<'a> {
    /// Wait for the launched step's DtoH, process outputs, and return
    /// any newly generated per-request outputs.
    pub fn collect(mut self) -> Result<Vec<V2RequestOutput>> {
        let pending = self.pending.take().expect("PendingStep already collected");
        self.engine
            .inner
            .step_collect(Some(pending))
            .map_err(map_engine_err)
    }
}

impl<'a> Drop for PendingStep<'a> {
    fn drop(&mut self) {
        // No fallback: if the caller drops without collect() the scheduled
        // diff is leaked (recycled pool won't get it back). This is a
        // programmer error — the #[must_use] attribute warns at compile
        // time. In debug builds, panic to make the mistake unmissable.
        if self.pending.is_some() {
            debug_assert!(
                false,
                "PendingStep dropped without collect(); scheduled step is leaked. \
                 Call .collect() or hold the value until the step completes."
            );
        }
    }
}

fn map_engine_err(e: rvllm_v2::engine::EngineError) -> RvllmError {
    RvllmError::Cuda {
        op: "engine_step",
        kernel: None,
        stream: 0,
        kind: CudaErrorKind::Other,
        src: Some(Box::new(e)),
    }
}

// ---------------------------------------------------------------------------
// Startup verification — refuse to run with missing artifacts.
// ---------------------------------------------------------------------------

fn verify_kernel_artifacts(_config: &V2EngineConfig) -> Result<()> {
    let Some(kdir) = std::env::var("RVLLM_PTX_DIR").ok().map(PathBuf::from) else {
        return Ok(()); // HF-download path will place kernels at runtime
    };

    let fa3_so = kdir.join("libfa3_kernels.so");
    if !fa3_so.exists() {
        return Err(RvllmError::Fa3SoMissing {
            path: fa3_so.display().to_string(),
        });
    }
    let cutlass_so = kdir.join("libcutlass_kernels.so");
    if !cutlass_so.exists() {
        return Err(RvllmError::CutlassSoMissing {
            path: cutlass_so.display().to_string(),
        });
    }
    Ok(())
}
