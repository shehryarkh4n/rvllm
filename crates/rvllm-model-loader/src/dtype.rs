use std::fmt;

/// Data types supported for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    U8,
    Q4_0,
    #[allow(non_camel_case_types)]
    Q8_0,
    #[allow(non_camel_case_types)]
    Q4_K_M,
    #[allow(non_camel_case_types)]
    Q5_K,
    #[allow(non_camel_case_types)]
    IQ4_NL,
    #[allow(non_camel_case_types)]
    IQ4_XS,
}

impl DType {
    /// Size in bytes of a single element for this dtype.
    /// Quantized types report 1 byte here; use `gguf_tensor_bytes()` for exact
    /// GGUF tensor sizing.
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I32 => 4,
            DType::U8 => 1,
            DType::Q4_0 | DType::Q8_0 | DType::Q4_K_M | DType::Q5_K | DType::IQ4_NL | DType::IQ4_XS => 1,
        }
    }

    /// Exact number of bytes occupied by a GGUF tensor with `numel` elements.
    pub fn gguf_tensor_bytes(&self, numel: usize) -> Option<usize> {
        fn exact_blocks(numel: usize, block: usize, bytes: usize) -> Option<usize> {
            if numel % block == 0 {
                Some((numel / block) * bytes)
            } else {
                None
            }
        }

        match self {
            DType::F32 => Some(numel * 4),
            DType::F16 | DType::BF16 => Some(numel * 2),
            DType::I32 => Some(numel * 4),
            DType::U8 => Some(numel),
            DType::Q4_0 => exact_blocks(numel, 32, 18),
            DType::Q8_0 => exact_blocks(numel, 32, 34),
            DType::Q4_K_M => exact_blocks(numel, 256, 144),
            DType::Q5_K => exact_blocks(numel, 256, 176),
            DType::IQ4_NL => exact_blocks(numel, 32, 18),
            DType::IQ4_XS => exact_blocks(numel, 256, 136),
        }
    }

    /// Parse from a safetensors dtype string.
    pub fn from_safetensors_str(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(DType::F32),
            "F16" => Some(DType::F16),
            "BF16" => Some(DType::BF16),
            "I32" => Some(DType::I32),
            "U8" | "BOOL" => Some(DType::U8),
            _ => None,
        }
    }

    /// Parse from a GGUF type code.
    pub fn from_gguf_type(code: u32) -> Option<Self> {
        match code {
            0 => Some(DType::F32),
            1 => Some(DType::F16),
            2 => Some(DType::Q4_0),
            8 => Some(DType::Q8_0),
            13 => Some(DType::Q5_K),
            20 => Some(DType::IQ4_NL),
            23 => Some(DType::IQ4_XS),
            26 => Some(DType::I32),
            30 => Some(DType::BF16),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "F32"),
            DType::F16 => write!(f, "F16"),
            DType::BF16 => write!(f, "BF16"),
            DType::I32 => write!(f, "I32"),
            DType::U8 => write!(f, "U8"),
            DType::Q4_0 => write!(f, "Q4_0"),
            DType::Q8_0 => write!(f, "Q8_0"),
            DType::Q4_K_M => write!(f, "Q4_K_M"),
            DType::Q5_K => write!(f, "Q5_K"),
            DType::IQ4_NL => write!(f, "IQ4_NL"),
            DType::IQ4_XS => write!(f, "IQ4_XS"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_of_standard_types() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F16.size_of(), 2);
        assert_eq!(DType::BF16.size_of(), 2);
        assert_eq!(DType::I32.size_of(), 4);
        assert_eq!(DType::U8.size_of(), 1);
    }

    #[test]
    fn from_safetensors_str_known() {
        assert_eq!(DType::from_safetensors_str("F32"), Some(DType::F32));
        assert_eq!(DType::from_safetensors_str("F16"), Some(DType::F16));
        assert_eq!(DType::from_safetensors_str("BF16"), Some(DType::BF16));
        assert_eq!(DType::from_safetensors_str("UNKNOWN"), None);
    }

    #[test]
    fn from_gguf_type_known() {
        assert_eq!(DType::from_gguf_type(0), Some(DType::F32));
        assert_eq!(DType::from_gguf_type(1), Some(DType::F16));
        assert_eq!(DType::from_gguf_type(2), Some(DType::Q4_0));
        assert_eq!(DType::from_gguf_type(255), None);
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", DType::Q4_K_M), "Q4_K_M");
        assert_eq!(format!("{}", DType::F16), "F16");
    }
}
