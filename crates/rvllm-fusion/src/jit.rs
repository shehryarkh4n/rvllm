use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::ir::FusedKernel;
#[cfg(feature = "llvm")]
use crate::llvm_backend::LlvmPtxCompiler;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum JitError {
    NvccNotFound,
    ArchDetectFailed(String),
    IoError(std::io::Error),
    CompilationFailed { stderr: String, exit_code: Option<i32> },
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::NvccNotFound => write!(f, "nvcc not found: checked $CUDA_HOME/bin/nvcc, /usr/local/cuda/bin/nvcc, and PATH"),
            JitError::ArchDetectFailed(msg) => write!(f, "GPU arch detection failed: {msg}"),
            JitError::IoError(e) => write!(f, "IO error: {e}"),
            JitError::CompilationFailed { stderr, exit_code } => {
                write!(f, "nvcc compilation failed (exit {:?}):\n{stderr}", exit_code)
            }
        }
    }
}

impl std::error::Error for JitError {}

impl From<std::io::Error> for JitError {
    fn from(e: std::io::Error) -> Self {
        JitError::IoError(e)
    }
}

pub type Result<T> = std::result::Result<T, JitError>;

// ---------------------------------------------------------------------------
// JitCompiler
// ---------------------------------------------------------------------------

pub struct JitCompiler {
    nvcc_path: PathBuf,
    cuda_arch: String,
    include_dirs: Vec<PathBuf>,
}

impl JitCompiler {
    /// Auto-detect nvcc location and GPU compute capability.
    pub fn new() -> Result<Self> {
        let nvcc_path = Self::find_nvcc()?;
        let cuda_arch = Self::detect_arch()?;
        tracing::info!("JIT compiler: nvcc={} arch={}", nvcc_path.display(), cuda_arch);
        Ok(Self {
            nvcc_path,
            cuda_arch,
            include_dirs: Vec::new(),
        })
    }

    /// Create with explicit settings.
    pub fn with_config(nvcc_path: PathBuf, cuda_arch: String, include_dirs: Vec<PathBuf>) -> Self {
        Self { nvcc_path, cuda_arch, include_dirs }
    }

    pub fn arch(&self) -> &str {
        &self.cuda_arch
    }

    pub fn nvcc_path(&self) -> &Path {
        &self.nvcc_path
    }

    pub fn add_include_dir(&mut self, dir: PathBuf) {
        self.include_dirs.push(dir);
    }

    /// Compile CUDA C source to PTX bytes.
    ///
    /// Thread-safe: each invocation uses its own temp directory.
    pub fn compile_to_ptx(&self, source: &str, kernel_name: &str) -> Result<Vec<u8>> {
        let t0 = Instant::now();

        // Unique temp dir per compilation (safe for parallel calls).
        let tmp_dir = std::env::temp_dir().join(format!(
            "rvllm_jit_{}_{:x}",
            kernel_name,
            // Mix thread id + instant for uniqueness without extra deps.
            {
                let tid = std::thread::current().id();
                let hash = t0.elapsed().as_nanos() as u64
                    ^ format!("{tid:?}").len() as u64
                    ^ std::process::id() as u64;
                hash
            }
        ));
        fs::create_dir_all(&tmp_dir)?;

        let src_path = tmp_dir.join(format!("{kernel_name}.cu"));
        let ptx_path = tmp_dir.join(format!("{kernel_name}.ptx"));

        // Write source.
        {
            let mut f = fs::File::create(&src_path)?;
            f.write_all(source.as_bytes())?;
        }

        // Build nvcc command.
        let mut cmd = Command::new(&self.nvcc_path);
        cmd.args([
            "-std=c++17",
            &format!("-arch={}", self.cuda_arch),
            "-ptx",
            "-O3",
            "--expt-relaxed-constexpr",
        ]);
        for dir in &self.include_dirs {
            cmd.arg("-I").arg(dir);
        }
        cmd.arg("-o").arg(&ptx_path);
        cmd.arg(&src_path);

        tracing::debug!("nvcc cmd: {:?}", cmd);

        let output = cmd.output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            // Clean up before returning error.
            let _ = fs::remove_dir_all(&tmp_dir);
            return Err(JitError::CompilationFailed {
                stderr,
                exit_code: output.status.code(),
            });
        }

        let ptx = fs::read(&ptx_path)?;

        // Clean up temp files.
        let _ = fs::remove_dir_all(&tmp_dir);

        let elapsed = t0.elapsed();
        tracing::info!(
            "JIT compiled '{}' -> {} bytes PTX in {:.1}ms",
            kernel_name,
            ptx.len(),
            elapsed.as_secs_f64() * 1000.0
        );

        Ok(ptx)
    }

    /// Detect GPU compute capability via nvidia-smi.
    /// Returns e.g. "sm_90", "sm_80".
    pub fn detect_arch() -> Result<String> {
        // Try nvidia-smi first.
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits", "--id=0"])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let cap = String::from_utf8_lossy(&o.stdout).trim().to_string();
                // cap is like "9.0" or "8.9" -- convert to "sm_90" / "sm_89"
                let parts: Vec<&str> = cap.split('.').collect();
                if parts.len() == 2 {
                    return Ok(format!("sm_{}{}", parts[0], parts[1]));
                }
                Err(JitError::ArchDetectFailed(format!("unexpected compute_cap format: '{cap}'")))
            }
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
                Err(JitError::ArchDetectFailed(format!("nvidia-smi failed: {stderr}")))
            }
            Err(e) => Err(JitError::ArchDetectFailed(format!("nvidia-smi not found: {e}"))),
        }
    }

    /// Locate nvcc binary.
    /// Search order: $CUDA_HOME/bin/nvcc, /usr/local/cuda/bin/nvcc, PATH.
    fn find_nvcc() -> Result<PathBuf> {
        // $CUDA_HOME/bin/nvcc
        if let Ok(home) = std::env::var("CUDA_HOME") {
            let p = Path::new(&home).join("bin/nvcc");
            if p.is_file() {
                return Ok(p);
            }
        }

        // /usr/local/cuda/bin/nvcc
        let default = PathBuf::from("/usr/local/cuda/bin/nvcc");
        if default.is_file() {
            return Ok(default);
        }

        // `which nvcc`
        if let Ok(output) = Command::new("which").arg("nvcc").output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(PathBuf::from(path));
                }
            }
        }

        Err(JitError::NvccNotFound)
    }
}

// ---------------------------------------------------------------------------
// LLVM-based compilation (no nvcc dependency)
// ---------------------------------------------------------------------------

#[cfg(feature = "llvm")]
impl JitCompiler {
    /// Compile a fused kernel to PTX via the LLVM NVPTX backend.
    /// Falls back to nvcc-based compilation if LLVM compilation fails.
    pub fn compile_via_llvm(
        &self,
        kernel: &FusedKernel,
        hidden: usize,
        out_dim: usize,
        eps: f32,
    ) -> Result<Vec<u8>> {
        let t0 = Instant::now();
        let compiler = LlvmPtxCompiler::new();
        let func_name = LlvmPtxCompiler::kernel_function_name(kernel);

        match compiler.compile_fused_kernel(kernel, hidden, out_dim, eps, &self.cuda_arch) {
            Ok(ptx) => {
                let elapsed = t0.elapsed();
                tracing::info!(
                    "LLVM compiled '{}' -> {} bytes PTX in {:.1}ms",
                    func_name,
                    ptx.len(),
                    elapsed.as_secs_f64() * 1000.0
                );
                Ok(ptx)
            }
            Err(e) => {
                tracing::warn!("LLVM compilation failed for '{}': {}, falling back to nvcc", func_name, e);
                // Fall back to CUDA C codegen + nvcc
                let cuda_src = crate::codegen::generate_cuda_source(kernel)
                    .ok_or_else(|| JitError::CompilationFailed {
                        stderr: format!("unsupported pattern for codegen fallback: {e}"),
                        exit_code: None,
                    })?;
                let cuda_name = crate::codegen::kernel_function_name(kernel);
                self.compile_to_ptx(&cuda_src, &cuda_name)
            }
        }
    }

    /// Compile via LLVM only, no nvcc fallback.
    pub fn compile_via_llvm_only(
        &self,
        kernel: &FusedKernel,
        hidden: usize,
        out_dim: usize,
        eps: f32,
    ) -> Result<Vec<u8>> {
        let compiler = LlvmPtxCompiler::new();
        compiler
            .compile_fused_kernel(kernel, hidden, out_dim, eps, &self.cuda_arch)
            .map_err(|e| JitError::CompilationFailed {
                stderr: e,
                exit_code: None,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_nvcc_doesnt_panic() {
        // Just verify it returns Ok or a sensible error, doesn't panic.
        let _ = JitCompiler::find_nvcc();
    }

    #[test]
    fn test_detect_arch_doesnt_panic() {
        let _ = JitCompiler::detect_arch();
    }

    /// If nvcc is available, do a real compile.
    #[test]
    fn test_compile_if_nvcc_available() {
        let compiler = match JitCompiler::new() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping JIT test: no nvcc / GPU");
                return;
            }
        };

        let source = r#"
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"#;

        let ptx = compiler.compile_to_ptx(source, "vector_add").unwrap();
        assert!(!ptx.is_empty());
        // PTX files start with a comment or directive containing ".version"
        let ptx_str = String::from_utf8_lossy(&ptx);
        assert!(ptx_str.contains(".version"), "output doesn't look like PTX");
    }

    #[test]
    fn test_compile_error_reports_stderr() {
        let compiler = match JitCompiler::new() {
            Ok(c) => c,
            Err(_) => return,
        };

        let bad_source = "this is not valid CUDA code @@@ !!!";
        let err = compiler.compile_to_ptx(bad_source, "bad_kernel").unwrap_err();
        match err {
            JitError::CompilationFailed { stderr, .. } => {
                assert!(!stderr.is_empty(), "expected nvcc stderr in error");
            }
            other => panic!("expected CompilationFailed, got: {other}"),
        }
    }
}
