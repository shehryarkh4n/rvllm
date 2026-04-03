//! SafeTensors GPU loader -- loads weights directly to CUDA device memory as f16.
//!
//! Memory-maps the safetensors file(s), parses the header to find tensor
//! metadata, then uploads each tensor's raw bytes to GPU as f16.
//!
//! All dtypes on disk (F16, BF16, F32) are converted to f16 at load time.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;

    use cudarc::driver::{CudaSlice, CudaStream};
    use memmap2::Mmap;
    use rvllm_core::error::{LLMError, Result};
    use tracing::{debug, info};

    use crate::dtype::DType;
    use crate::gguf::GGUFLoader;
    use crate::weights::MockGpuAllocator;
    use crate::ModelFormat;

    /// Load all safetensors weights as f16 on GPU.
    ///
    /// F16 weights are uploaded directly (zero conversion), BF16 are converted
    /// to f16 on the host, and F32 weights are narrowed to f16.
    pub fn load_weights_to_gpu(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<HashMap<String, CudaSlice<half::f16>>> {
        let (weights, _shapes) = load_weights_to_gpu_with_shapes(path, stream)?;
        Ok(weights)
    }

    pub fn load_weights_to_gpu_with_shapes(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(HashMap<String, CudaSlice<half::f16>>, HashMap<String, Vec<usize>>)> {
        match crate::detect_format(path)? {
            ModelFormat::SafeTensors => {
                if path.is_dir() {
                    load_sharded_to_gpu(path, stream)
                } else {
                    load_single_to_gpu(path, stream)
                }
            }
            ModelFormat::GGUF => load_gguf_to_gpu(path, stream),
        }
    }

    fn load_single_to_gpu(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(HashMap<String, CudaSlice<half::f16>>, HashMap<String, Vec<usize>>)> {
        info!("gpu_loader: memory-mapping {}", path.display());

        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            LLMError::ModelError(format!("mmap failed for {}: {}", path.display(), e))
        })?;
        let data: &[u8] = &mmap;

        let (header, data_start) = parse_safetensors_header(data, path)?;

        let mut weights: HashMap<String, CudaSlice<half::f16>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        for (name, meta) in &header {
            if name == "__metadata__" {
                continue;
            }

            let (dtype_str, shape, tensor_bytes) =
                parse_tensor_meta(meta, name, data, data_start)?;
            let numel: usize = shape.iter().product();

            let f16_host = convert_to_f16(tensor_bytes, dtype_str, numel, name)?;

            let gpu_slice = stream.clone_htod(&f16_host).map_err(|e| {
                LLMError::GpuError(format!(
                    "clone_htod failed for tensor {} ({} f16 elems): {}",
                    name,
                    f16_host.len(),
                    e
                ))
            })?;

            debug!(
                tensor = name.as_str(),
                dtype = dtype_str,
                shape = ?shape,
                numel = numel,
                "uploaded tensor to GPU (f16)"
            );

            shapes.insert(name.clone(), shape.clone());
            weights.insert(name.clone(), gpu_slice);
        }

        info!(
            "gpu_loader: loaded {} tensors from {} to GPU (f16)",
            weights.len(),
            path.display()
        );
        Ok((weights, shapes))
    }

    fn load_sharded_to_gpu(
        dir: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(HashMap<String, CudaSlice<half::f16>>, HashMap<String, Vec<usize>>)> {
        let shard_files = collect_shards(dir)?;

        info!(
            "gpu_loader: loading {} shards from {} to GPU (f16)",
            shard_files.len(),
            dir.display()
        );

        let mut all_weights: HashMap<String, CudaSlice<half::f16>> = HashMap::new();
        let mut all_shapes: HashMap<String, Vec<usize>> = HashMap::new();
        for shard_path in &shard_files {
            let (shard_weights, shard_shapes) = load_single_to_gpu(shard_path, stream)?;
            all_weights.extend(shard_weights);
            all_shapes.extend(shard_shapes);
        }

        info!(
            "gpu_loader: loaded {} total tensors from {} shards (f16)",
            all_weights.len(),
            shard_files.len()
        );
        Ok((all_weights, all_shapes))
    }

    fn load_gguf_to_gpu(
        path: &Path,
        stream: &Arc<CudaStream>,
    ) -> Result<(HashMap<String, CudaSlice<half::f16>>, HashMap<String, Vec<usize>>)> {
        let gguf_path = if path.is_dir() {
            std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .find(|e| e.path().extension().map(|ext| ext == "gguf").unwrap_or(false))
                .map(|e| e.path())
                .ok_or_else(|| LLMError::ModelError(format!("no .gguf file found in {}", path.display())))?
        } else {
            path.to_path_buf()
        };

        info!("gpu_loader: reading GGUF {}", gguf_path.display());
        let raw_weights = GGUFLoader::load(&gguf_path, &MockGpuAllocator)?;
        let mut weights: HashMap<String, CudaSlice<half::f16>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        let names: Vec<String> = raw_weights.names().map(str::to_string).collect();
        for name in names {
            let tensor = raw_weights
                .get(&name)
                .ok_or_else(|| LLMError::ModelError(format!("GGUF tensor disappeared during load: {}", name)))?;
            let f16_host = convert_gguf_tensor_to_f16(
                tensor.data().as_bytes(),
                tensor.dtype(),
                tensor.shape(),
                &name,
            )?;
            let gpu_slice = stream.clone_htod(&f16_host).map_err(|e| {
                LLMError::GpuError(format!(
                    "clone_htod failed for GGUF tensor {} ({} f16 elems): {}",
                    name,
                    f16_host.len(),
                    e
                ))
            })?;
            let shape = tensor.shape().to_vec();
            shapes.insert(name.clone(), shape.clone());
            weights.insert(name.clone(), gpu_slice);
            maybe_insert_gguf_aliases(&name, &shape, &mut shapes, &mut weights);
        }

        info!(
            "gpu_loader: loaded {} GGUF tensors from {} to GPU (f16 path)",
            weights.len(),
            gguf_path.display()
        );
        Ok((weights, shapes))
    }

    fn maybe_insert_gguf_aliases(
        name: &str,
        shape: &[usize],
        shapes: &mut HashMap<String, Vec<usize>>,
        weights: &mut HashMap<String, CudaSlice<half::f16>>,
    ) {
        let alias = match name {
            "token_embd.weight" => Some("model.embed_tokens.weight"),
            "output_norm.weight" => Some("model.norm.weight"),
            "output.weight" => Some("lm_head.weight"),
            _ => None,
        };

        if let Some(alias) = alias {
            if !weights.contains_key(alias) {
                if let Some(buf) = weights.get(name).cloned() {
                    weights.insert(alias.to_string(), buf);
                    shapes.insert(alias.to_string(), shape.to_vec());
                }
            }
        }
    }

    const KVALUES_IQ4_NL: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
        1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
    ];

    fn dequantize_iq4_nl(bytes: &[u8], numel: usize, tensor_name: &str) -> Result<Vec<half::f16>> {
        const BLOCK: usize = 32;
        const BYTES_PER_BLOCK: usize = 18;
        if numel % BLOCK != 0 {
            return Err(LLMError::ModelError(format!(
                "GGUF tensor {} IQ4_NL element count {} is not divisible by {}",
                tensor_name, numel, BLOCK
            )));
        }
        let nblocks = numel / BLOCK;
        if bytes.len() != nblocks * BYTES_PER_BLOCK {
            return Err(LLMError::ModelError(format!(
                "GGUF tensor {} IQ4_NL size mismatch: {} bytes for {} blocks",
                tensor_name, bytes.len(), nblocks
            )));
        }
        let mut out = Vec::with_capacity(numel);
        for block in bytes.chunks_exact(BYTES_PER_BLOCK) {
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let qs = &block[2..18];
            for &q in qs {
                out.push(half::f16::from_f32(d * KVALUES_IQ4_NL[(q & 0x0F) as usize]));
            }
            for &q in qs {
                out.push(half::f16::from_f32(d * KVALUES_IQ4_NL[(q >> 4) as usize]));
            }
        }
        Ok(out)
    }

    fn dequantize_iq4_xs(bytes: &[u8], numel: usize, tensor_name: &str) -> Result<Vec<half::f16>> {
        const SUPER_BLOCK: usize = 256;
        const BYTES_PER_BLOCK: usize = 136;
        if numel % SUPER_BLOCK != 0 {
            return Err(LLMError::ModelError(format!(
                "GGUF tensor {} IQ4_XS element count {} is not divisible by {}",
                tensor_name, numel, SUPER_BLOCK
            )));
        }
        let nblocks = numel / SUPER_BLOCK;
        if bytes.len() != nblocks * BYTES_PER_BLOCK {
            return Err(LLMError::ModelError(format!(
                "GGUF tensor {} IQ4_XS size mismatch: {} bytes for {} blocks",
                tensor_name, bytes.len(), nblocks
            )));
        }
        let mut out = Vec::with_capacity(numel);
        for block in bytes.chunks_exact(BYTES_PER_BLOCK) {
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let scales_h = u16::from_le_bytes([block[2], block[3]]);
            let scales_l = &block[4..8];
            let qs = &block[8..136];
            for ib in 0..(SUPER_BLOCK / 32) {
                let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as u16
                    | (((scales_h >> (2 * ib)) & 0x03) << 4);
                let dl = d * (ls as f32 - 32.0);
                let group = &qs[ib * 16..(ib + 1) * 16];
                for &q in group {
                    out.push(half::f16::from_f32(dl * KVALUES_IQ4_NL[(q & 0x0F) as usize]));
                }
                for &q in group {
                    out.push(half::f16::from_f32(dl * KVALUES_IQ4_NL[(q >> 4) as usize]));
                }
            }
        }
        Ok(out)
    }

    fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }

    fn dequantize_q5_k(bytes: &[u8], numel: usize, tensor_name: &str) -> Result<Vec<half::f16>> {
        const SUPER_BLOCK: usize = 256;
        const BYTES_PER_BLOCK: usize = 176;
        if numel % SUPER_BLOCK != 0 {
            return Err(LLMError::ModelError(format!(
                "GGUF tensor {} Q5_K element count {} is not divisible by {}",
                tensor_name, numel, SUPER_BLOCK
            )));
        }
        let nblocks = numel / SUPER_BLOCK;
        if bytes.len() != nblocks * BYTES_PER_BLOCK {
            return Err(LLMError::ModelError(format!(
                "GGUF tensor {} Q5_K size mismatch: {} bytes for {} blocks",
                tensor_name, bytes.len(), nblocks
            )));
        }
        let mut out = Vec::with_capacity(numel);
        for block in bytes.chunks_exact(BYTES_PER_BLOCK) {
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
            let scales = &block[4..16];
            let qh = &block[16..48];
            let qs = &block[48..176];

            for j in 0..8 {
                let (sc_u8, m_u8) = get_scale_min_k4(j, scales);
                let sc = d * (sc_u8 as f32);
                let min = dmin * (m_u8 as f32);
                let ql = &qs[j * 16..(j + 1) * 16];
                let qh_block = &qh[j * 4..(j + 1) * 4];

                for l in 0..16 {
                    let hb = ((qh_block[l / 8] >> (l % 8)) & 1) << 4;
                    let q = ((ql[l] & 0x0F) | hb) as f32;
                    out.push(half::f16::from_f32(sc * q - min));
                }
                for l in 0..16 {
                    let idx = l + 16;
                    let hb = ((qh_block[idx / 8] >> (idx % 8)) & 1) << 4;
                    let q = (((ql[l] >> 4) & 0x0F) | hb) as f32;
                    out.push(half::f16::from_f32(sc * q - min));
                }
            }
        }
        Ok(out)
    }

    fn dequantize_q8_0(bytes: &[u8], numel: usize, tensor_name: &str) -> Result<Vec<half::f16>> {
        const BLOCK: usize = 32;
        const BYTES_PER_BLOCK: usize = 34;
        if numel % BLOCK != 0 {
            return Err(LLMError::ModelError(format!(
                "GGUF tensor {} Q8_0 element count {} is not divisible by {}",
                tensor_name, numel, BLOCK
            )));
        }
        let nblocks = numel / BLOCK;
        if bytes.len() != nblocks * BYTES_PER_BLOCK {
            return Err(LLMError::ModelError(format!(
                "GGUF tensor {} Q8_0 size mismatch: {} bytes for {} blocks",
                tensor_name, bytes.len(), nblocks
            )));
        }
        let mut out = Vec::with_capacity(numel);
        for block in bytes.chunks_exact(BYTES_PER_BLOCK) {
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            for &q in &block[2..34] {
                let v = (q as i8) as f32;
                out.push(half::f16::from_f32(d * v));
            }
        }
        Ok(out)
    }

    fn convert_gguf_tensor_to_f16(
        bytes: &[u8],
        dtype: DType,
        shape: &[usize],
        tensor_name: &str,
    ) -> Result<Vec<half::f16>> {
        let numel = if shape.is_empty() { 1 } else { shape.iter().product() };
        match dtype {
            DType::F16 => {
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "GGUF tensor {} F16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect())
            }
            DType::BF16 => {
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "GGUF tensor {} BF16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|chunk| half::f16::from_f32(half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()))
                    .collect())
            }
            DType::F32 => {
                if bytes.len() != numel * 4 {
                    return Err(LLMError::ModelError(format!(
                        "GGUF tensor {} F32 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| half::f16::from_f32(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
                    .collect())
            }
            DType::I32 => {
                if bytes.len() != numel * 4 {
                    return Err(LLMError::ModelError(format!(
                        "GGUF tensor {} I32 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| half::f16::from_f32(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32))
                    .collect())
            }
            DType::U8 => Ok(bytes.iter().map(|&v| half::f16::from_f32(v as f32)).collect()),
            DType::Q8_0 => dequantize_q8_0(bytes, numel, tensor_name),
            DType::Q5_K => dequantize_q5_k(bytes, numel, tensor_name),
            DType::IQ4_NL => dequantize_iq4_nl(bytes, numel, tensor_name),
            DType::IQ4_XS => dequantize_iq4_xs(bytes, numel, tensor_name),
            DType::Q4_0 | DType::Q4_K_M => Err(LLMError::ModelError(format!(
                "gpu_loader: GGUF tensor '{}' uses quantized dtype {} which is not yet supported by the live CUDA f16 upload path",
                tensor_name, dtype
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // Shared helpers
    // -----------------------------------------------------------------------

    fn parse_safetensors_header(
        data: &[u8],
        path: &Path,
    ) -> Result<(HashMap<String, serde_json::Value>, usize)> {
        if data.len() < 8 {
            return Err(LLMError::ModelError(
                "safetensors file too small for header".into(),
            ));
        }

        let header_size = u64::from_le_bytes(
            data[..8]
                .try_into()
                .map_err(|_| LLMError::ModelError("invalid header size bytes".into()))?,
        ) as usize;

        if 8 + header_size > data.len() {
            return Err(LLMError::ModelError(
                "header size exceeds file length".into(),
            ));
        }

        let header_bytes = &data[8..8 + header_size];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|e| LLMError::ModelError(format!("invalid header utf8: {}", e)))?;
        let header: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| LLMError::SerializationError(format!("header json: {}", e)))?;

        Ok((header, 8 + header_size))
    }

    fn parse_tensor_meta<'a, 'b>(
        meta: &'b serde_json::Value,
        name: &str,
        data: &'a [u8],
        data_start: usize,
    ) -> Result<(&'b str, Vec<usize>, &'a [u8])> {
        let obj = meta.as_object().ok_or_else(|| {
            LLMError::ModelError(format!("tensor {} has non-object meta", name))
        })?;

        let dtype_str = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing dtype", name)))?;

        let shape: Vec<usize> = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| LLMError::ModelError(format!("tensor {} missing shape", name)))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| LLMError::ModelError("invalid shape element".into()))
            })
            .collect::<Result<Vec<_>>>()?;

        let offsets = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                LLMError::ModelError(format!("tensor {} missing data_offsets", name))
            })?;

        if offsets.len() != 2 {
            return Err(LLMError::ModelError(format!(
                "tensor {} has {} offsets, expected 2",
                name,
                offsets.len()
            )));
        }

        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;
        let abs_start = data_start + start;
        let abs_end = data_start + end;

        if abs_end > data.len() {
            return Err(LLMError::ModelError(format!(
                "tensor {} data range [{}, {}) exceeds file size {}",
                name,
                abs_start,
                abs_end,
                data.len()
            )));
        }

        Ok((dtype_str, shape, &data[abs_start..abs_end]))
    }

    fn collect_shards(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
        let mut shard_files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();
        shard_files.sort();

        if shard_files.is_empty() {
            return Err(LLMError::ModelError(format!(
                "no .safetensors files found in {}",
                dir.display()
            )));
        }
        Ok(shard_files)
    }

    /// Convert raw tensor bytes to `Vec<half::f16>`.
    ///
    /// - F16: reinterpret bytes directly (zero conversion).
    /// - BF16: convert bf16 -> f16 via f32 intermediate.
    /// - F32: narrow f32 -> f16.
    fn convert_to_f16(
        bytes: &[u8],
        dtype_str: &str,
        numel: usize,
        tensor_name: &str,
    ) -> Result<Vec<half::f16>> {
        match dtype_str {
            "F16" => {
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} F16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                let mut out = vec![half::f16::ZERO; numel];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr() as *mut u8,
                        bytes.len(),
                    );
                }
                Ok(out)
            }
            "BF16" => {
                if bytes.len() != numel * 2 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} BF16 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                #[cfg(feature = "zig")]
                {
                    let src = unsafe {
                        std::slice::from_raw_parts(bytes.as_ptr() as *const u16, numel)
                    };
                    let mut out = vec![half::f16::ZERO; numel];
                    rvllm_zig::bf16_to_f16(src, unsafe {
                        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, numel)
                    });
                    Ok(out)
                }
                #[cfg(not(feature = "zig"))]
                {
                    let mut out = Vec::with_capacity(numel);
                    for i in 0..numel {
                        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                        let bf = half::bf16::from_bits(bits);
                        out.push(half::f16::from_f32(bf.to_f32()));
                    }
                    Ok(out)
                }
            }
            "F32" => {
                if bytes.len() != numel * 4 {
                    return Err(LLMError::ModelError(format!(
                        "tensor {} F32 size mismatch: {} bytes for {} elements",
                        tensor_name,
                        bytes.len(),
                        numel
                    )));
                }
                let src =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, numel) };
                #[cfg(feature = "zig")]
                {
                    let mut out = vec![half::f16::ZERO; numel];
                    rvllm_zig::f32_to_f16(src, unsafe {
                        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, numel)
                    });
                    Ok(out)
                }
                #[cfg(not(feature = "zig"))]
                {
                    let mut out = Vec::with_capacity(numel);
                    for &v in src {
                        out.push(half::f16::from_f32(v));
                    }
                    Ok(out)
                }
            }
            _ => Err(LLMError::ModelError(format!(
                "gpu_loader: unsupported dtype '{}' for tensor '{}', only F32/F16/BF16 supported",
                dtype_str, tensor_name
            ))),
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::{load_weights_to_gpu, load_weights_to_gpu_with_shapes};

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        assert!(true);
    }
}
