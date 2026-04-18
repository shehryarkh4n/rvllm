#![cfg(feature = "tpu")]

//! Loads exported StableHLO artifacts for PJRT compilation.
//!
//! An artifact directory (produced by `gemma4_export.py`) contains:
//!   model.mlir          -- StableHLO MLIR text or bytecode
//!   manifest.json       -- input/output specs, donate indices, metadata
//!   compile_options.pb  -- (optional) serialized xla::CompileOptionsProto

use std::path::Path;

use serde::Deserialize;
use tracing::info;

use crate::client::{CompiledExecutable, PjrtClientHandle};
use crate::ffi::PjrtElementType;
use crate::{LLMError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardingAxis {
    /// Tensor is sharded (split) along this axis for tensor parallelism.
    Axis(usize),
    /// Tensor is replicated across all devices.
    Replicated,
}

#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub shape: Vec<i64>,
    pub dtype: PjrtElementType,
    pub sharding: Option<ShardingAxis>,
}

#[derive(Debug)]
pub struct ExportedModel {
    pub mlir_bytes: Vec<u8>,
    pub input_specs: Vec<TensorSpec>,
    pub output_specs: Vec<TensorSpec>,
    pub donate_indices: Vec<usize>,
    /// Optional pre-serialized CompileOptionsProto (from compile_options.pb).
    pub compile_options: Option<Vec<u8>>,
    /// Number of SPMD partitions inferred from sharding specs.
    pub num_partitions: usize,
}

// ---------------------------------------------------------------------------
// Manifest JSON schema (what gemma4_export.py writes)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Manifest {
    /// Filename of the MLIR module (e.g. "model.mlir").
    #[serde(default = "default_mlir_file")]
    mlir_file: String,
    inputs: Vec<ManifestTensor>,
    outputs: Vec<ManifestTensor>,
    #[serde(default)]
    donate_indices: Vec<usize>,
    #[serde(default)]
    num_partitions: Option<usize>,
}

fn default_mlir_file() -> String {
    "model.mlir".into()
}

#[derive(Deserialize)]
struct ManifestTensor {
    shape: Vec<i64>,
    dtype: String,
    #[serde(default)]
    sharding: Option<ManifestSharding>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ManifestSharding {
    Axis(usize),
    Named(String),
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load an exported StableHLO artifact directory.
///
/// Expected layout:
/// ```text
/// dir/
///   manifest.json
///   model.mlir            (or whatever mlir_file says)
///   compile_options.pb    (optional)
/// ```
pub fn load_exported_model(dir: &Path) -> Result<ExportedModel> {
    let manifest_path = dir.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|e| {
        LLMError::GpuError(format!(
            "failed to read {}: {e}",
            manifest_path.display()
        ))
    })?;

    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).map_err(|e| {
        LLMError::GpuError(format!(
            "invalid manifest.json at {}: {e}",
            manifest_path.display()
        ))
    })?;

    let mlir_path = dir.join(&manifest.mlir_file);
    let mlir_bytes = std::fs::read(&mlir_path).map_err(|e| {
        LLMError::GpuError(format!(
            "failed to read MLIR at {}: {e}",
            mlir_path.display()
        ))
    })?;

    if mlir_bytes.is_empty() {
        return Err(LLMError::GpuError(format!(
            "MLIR file {} is empty",
            mlir_path.display()
        )));
    }

    let input_specs: Vec<TensorSpec> = manifest
        .inputs
        .iter()
        .map(convert_tensor_spec)
        .collect::<Result<_>>()?;

    let output_specs: Vec<TensorSpec> = manifest
        .outputs
        .iter()
        .map(convert_tensor_spec)
        .collect::<Result<_>>()?;

    // Validate donate indices are in range
    for &idx in &manifest.donate_indices {
        if idx >= input_specs.len() {
            return Err(LLMError::GpuError(format!(
                "donate_index {} out of range (have {} inputs)",
                idx,
                input_specs.len()
            )));
        }
    }

    // Infer num_partitions from sharding specs if not explicitly set
    let num_partitions = manifest.num_partitions.unwrap_or_else(|| {
        infer_num_partitions(&input_specs)
    });

    // Load compile_options.pb if present
    let opts_path = dir.join("compile_options.pb");
    let compile_options = if opts_path.exists() {
        Some(std::fs::read(&opts_path).map_err(|e| {
            LLMError::GpuError(format!(
                "failed to read {}: {e}",
                opts_path.display()
            ))
        })?)
    } else {
        None
    };

    info!(
        mlir_file = %manifest.mlir_file,
        mlir_bytes = mlir_bytes.len(),
        inputs = input_specs.len(),
        outputs = output_specs.len(),
        donated = manifest.donate_indices.len(),
        num_partitions,
        "loaded exported model from {}",
        dir.display()
    );

    Ok(ExportedModel {
        mlir_bytes,
        input_specs,
        output_specs,
        donate_indices: manifest.donate_indices,
        compile_options,
        num_partitions,
    })
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

/// Compile an exported model to a PJRT executable.
///
/// If the model has custom compile_options (from compile_options.pb), those
/// are applied to the client before compilation. Otherwise, the default
/// compile options are used with num_partitions set from the sharding specs.
pub fn compile_model(
    client: &mut PjrtClientHandle,
    model: &ExportedModel,
) -> Result<CompiledExecutable> {
    // Apply compile options: prefer the model's bundled options, otherwise
    // build a minimal CompileOptionsProto with the right num_partitions.
    let opts = match &model.compile_options {
        Some(pb) => pb.clone(),
        None => build_compile_options(model.num_partitions),
    };
    client.set_compile_options(opts);

    info!(
        mlir_bytes = model.mlir_bytes.len(),
        num_partitions = model.num_partitions,
        "compiling StableHLO model"
    );

    // Detect whether this is MLIR text or bytecode.
    // MLIR bytecode starts with "ML\xefR" magic bytes.
    let is_bytecode = model.mlir_bytes.len() >= 4
        && model.mlir_bytes[0] == b'M'
        && model.mlir_bytes[1] == b'L'
        && model.mlir_bytes[2] == 0xEF
        && model.mlir_bytes[3] == b'R';

    if is_bytecode {
        client.compile_bytecode(&model.mlir_bytes)
    } else {
        // Text MLIR -- compile_bytes handles both, the PJRT format tag is "mlir"
        // for both text and bytecode.
        client.compile_bytecode(&model.mlir_bytes)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn convert_tensor_spec(mt: &ManifestTensor) -> Result<TensorSpec> {
    let dtype = parse_dtype(&mt.dtype)?;
    let sharding = mt.sharding.as_ref().map(|s| match s {
        ManifestSharding::Axis(a) => ShardingAxis::Axis(*a),
        ManifestSharding::Named(name) if name == "replicated" => ShardingAxis::Replicated,
        ManifestSharding::Named(_) => ShardingAxis::Replicated,
    });
    Ok(TensorSpec {
        shape: mt.shape.clone(),
        dtype,
        sharding,
    })
}

fn parse_dtype(s: &str) -> Result<PjrtElementType> {
    match s {
        "float32" | "f32" => Ok(PjrtElementType::F32),
        "float16" | "f16" => Ok(PjrtElementType::F16),
        "bfloat16" | "bf16" => Ok(PjrtElementType::BF16),
        "float64" | "f64" => Ok(PjrtElementType::F64),
        "int32" | "i32" => Ok(PjrtElementType::S32),
        "int64" | "i64" => Ok(PjrtElementType::S64),
        "int16" | "i16" => Ok(PjrtElementType::S16),
        "int8" | "i8" => Ok(PjrtElementType::S8),
        "uint32" | "u32" => Ok(PjrtElementType::U32),
        "uint16" | "u16" => Ok(PjrtElementType::U16),
        "uint8" | "u8" => Ok(PjrtElementType::U8),
        "bool" => Ok(PjrtElementType::PRED),
        "float8_e5m2" | "f8e5m2" => Ok(PjrtElementType::F8E5M2),
        "float8_e4m3fn" | "f8e4m3fn" => Ok(PjrtElementType::F8E4M3FN),
        _ => Err(LLMError::GpuError(format!(
            "unknown dtype in manifest: '{s}'"
        ))),
    }
}

/// Infer the number of SPMD partitions from sharding annotations.
/// If any input is sharded, we need at least as many partitions as the model
/// expects. Without explicit info, default to 1 (no partitioning).
fn infer_num_partitions(specs: &[TensorSpec]) -> usize {
    // If any spec has a sharding axis, assume multi-device; otherwise 1.
    let has_sharding = specs
        .iter()
        .any(|s| matches!(s.sharding, Some(ShardingAxis::Axis(_))));
    if has_sharding { 2 } else { 1 }
}

/// Build a minimal serialized xla::CompileOptionsProto.
///
/// The wire format (protobuf) for the fields we care about:
///   ExecutableBuildOptionsProto (field 2 in CompileOptionsProto):
///     num_partitions (field 3): varint
///     num_replicas   (field 4): varint
///     use_spmd_partitioning (field 7): varint (bool)
///
/// We hand-encode this to avoid pulling in a protobuf crate.
fn build_compile_options(num_partitions: usize) -> Vec<u8> {
    let mut inner = Vec::new();

    // field 3 (num_replicas) = 1
    // tag = (3 << 3) | 0 = 24
    inner.push(24);
    inner.push(1);

    // field 4 (num_partitions) = num_partitions
    // tag = (4 << 3) | 0 = 32
    inner.push(32);
    encode_varint(num_partitions as u64, &mut inner);

    if num_partitions > 1 {
        // field 7 (use_spmd_partitioning) = true
        // tag = (7 << 3) | 0 = 56
        inner.push(56);
        inner.push(1);
    }

    let mut outer = Vec::new();
    // field 2 (executable_build_options), wire type 2 (length-delimited)
    // tag = (2 << 3) | 2 = 18
    outer.push(18);
    encode_varint(inner.len() as u64, &mut outer);
    outer.extend_from_slice(&inner);

    outer
}

fn encode_varint(mut val: u64, buf: &mut Vec<u8>) {
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dtype_variants() {
        assert_eq!(parse_dtype("float32").unwrap(), PjrtElementType::F32);
        assert_eq!(parse_dtype("bf16").unwrap(), PjrtElementType::BF16);
        assert_eq!(parse_dtype("bfloat16").unwrap(), PjrtElementType::BF16);
        assert_eq!(parse_dtype("int32").unwrap(), PjrtElementType::S32);
        assert_eq!(parse_dtype("bool").unwrap(), PjrtElementType::PRED);
        assert_eq!(parse_dtype("float8_e4m3fn").unwrap(), PjrtElementType::F8E4M3FN);
        assert!(parse_dtype("complex128").is_err());
    }

    #[test]
    fn varint_encoding() {
        let mut buf = Vec::new();
        encode_varint(1, &mut buf);
        assert_eq!(buf, vec![1]);

        buf.clear();
        encode_varint(300, &mut buf);
        assert_eq!(buf, vec![0xAC, 0x02]);

        buf.clear();
        encode_varint(0, &mut buf);
        assert_eq!(buf, vec![0]);
    }

    #[test]
    fn build_compile_options_single() {
        let opts = build_compile_options(1);
        // Should contain field 2 (embedded message) with num_replicas=1, num_partitions=1
        assert!(!opts.is_empty());
        // Should NOT contain use_spmd_partitioning
        assert!(!opts.contains(&56));
    }

    #[test]
    fn build_compile_options_multi() {
        let opts = build_compile_options(4);
        assert!(!opts.is_empty());
        // Should contain use_spmd_partitioning=true (tag 56)
        assert!(opts.contains(&56));
    }

    #[test]
    fn infer_partitions_no_sharding() {
        let specs = vec![
            TensorSpec {
                shape: vec![128, 4096],
                dtype: PjrtElementType::BF16,
                sharding: None,
            },
        ];
        assert_eq!(infer_num_partitions(&specs), 1);
    }

    #[test]
    fn infer_partitions_with_sharding() {
        let specs = vec![
            TensorSpec {
                shape: vec![128, 4096],
                dtype: PjrtElementType::BF16,
                sharding: Some(ShardingAxis::Axis(1)),
            },
        ];
        assert_eq!(infer_num_partitions(&specs), 2);
    }

    #[test]
    fn infer_partitions_replicated_only() {
        let specs = vec![
            TensorSpec {
                shape: vec![128, 4096],
                dtype: PjrtElementType::BF16,
                sharding: Some(ShardingAxis::Replicated),
            },
        ];
        assert_eq!(infer_num_partitions(&specs), 1);
    }

    #[test]
    fn load_manifest_json() {
        let json = r#"{
            "mlir_file": "gemma4.mlir",
            "inputs": [
                {"shape": [1, 128], "dtype": "int32"},
                {"shape": [32, 16, 8, 256], "dtype": "bfloat16", "sharding": 0},
                {"shape": [32, 16, 8, 256], "dtype": "bfloat16", "sharding": 0}
            ],
            "outputs": [
                {"shape": [1, 256000], "dtype": "float32"}
            ],
            "donate_indices": [1, 2],
            "num_partitions": 4
        }"#;
        let m: Manifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.mlir_file, "gemma4.mlir");
        assert_eq!(m.inputs.len(), 3);
        assert_eq!(m.outputs.len(), 1);
        assert_eq!(m.donate_indices, vec![1, 2]);
        assert_eq!(m.num_partitions, Some(4));

        let specs: Vec<TensorSpec> = m
            .inputs
            .iter()
            .map(|t| convert_tensor_spec(t).unwrap())
            .collect();
        assert_eq!(specs[0].dtype, PjrtElementType::S32);
        assert_eq!(specs[0].sharding, None);
        assert_eq!(specs[1].sharding, Some(ShardingAxis::Axis(0)));
    }

    #[test]
    fn load_manifest_defaults() {
        let json = r#"{
            "inputs": [{"shape": [8], "dtype": "f32"}],
            "outputs": [{"shape": [8], "dtype": "f32"}]
        }"#;
        let m: Manifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.mlir_file, "model.mlir");
        assert!(m.donate_indices.is_empty());
        assert_eq!(m.num_partitions, None);
    }

    #[test]
    fn load_exported_model_from_dir() {
        let tmp = std::env::temp_dir().join("rvllm_artifact_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let mlir = r#"module @jit_test {
  func.func public @main(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> (tensor<8xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}"#;
        std::fs::write(tmp.join("model.mlir"), mlir).unwrap();

        let manifest = r#"{
            "inputs": [
                {"shape": [8], "dtype": "float32"},
                {"shape": [8], "dtype": "float32"}
            ],
            "outputs": [
                {"shape": [8], "dtype": "float32"}
            ],
            "donate_indices": [1]
        }"#;
        std::fs::write(tmp.join("manifest.json"), manifest).unwrap();

        let model = load_exported_model(&tmp).unwrap();
        assert_eq!(model.input_specs.len(), 2);
        assert_eq!(model.output_specs.len(), 1);
        assert_eq!(model.donate_indices, vec![1]);
        assert_eq!(model.num_partitions, 1);
        assert!(model.compile_options.is_none());
        assert!(!model.mlir_bytes.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn load_bad_donate_index() {
        let tmp = std::env::temp_dir().join("rvllm_artifact_bad_donate");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        std::fs::write(tmp.join("model.mlir"), "module {}").unwrap();
        let manifest = r#"{
            "inputs": [{"shape": [8], "dtype": "f32"}],
            "outputs": [{"shape": [8], "dtype": "f32"}],
            "donate_indices": [5]
        }"#;
        std::fs::write(tmp.join("manifest.json"), manifest).unwrap();

        let err = load_exported_model(&tmp).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("donate_index 5 out of range"), "got: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
