//! Per-request sampling parameters.
//!
//! Temperature = 0.0 means greedy (argmax only). Otherwise the
//! top-k/top-p sampler is invoked; that kernel is optional and
//! currently a stub.

#[derive(Copy, Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub seed: u64,
}

impl SamplingParams {
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: 0,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 && self.top_k.is_none() && self.top_p.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_greedy() {
        assert!(SamplingParams::greedy().is_greedy());
    }

    #[test]
    fn nonzero_temp_is_not_greedy() {
        let mut p = SamplingParams::greedy();
        p.temperature = 0.7;
        assert!(!p.is_greedy());
    }
}
