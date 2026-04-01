/// Shapes extracted from a model config, enough to pre-compile all fused kernels.
#[derive(Debug, Clone)]
pub struct ModelShapes {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
}
