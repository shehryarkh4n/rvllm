//! rvllm-sampling: GPU-side greedy argmax + optional top-k/p.
//!
//! DtoH coordination uses a consume-once `DtoHTicket<'p>` that borrows
//! `&mut PinnedTokens`. The type state makes "launch twice before
//! collect" and "read without wait" compile errors.

pub mod dtoh;
pub mod params;

pub use dtoh::{DtoHTicket, PinnedTokens};
pub use params::SamplingParams;
