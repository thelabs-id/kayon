use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

// "GGUF" (0x47 0x47 0x55 0x46) read as a little-endian u32.
const GGUF_MAGIC: u32 = 0x4655_4747;

#[derive(Clone, Debug, Default)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

#[derive(Clone, Debug)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U16(v) => Some(*v as u32),
            GgufValue::U8(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U64(v) => Some(*v),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::I64(v) => Some(*v as u64),
            GgufValue::I32(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_u16_le<R: Read>(r: &mut R) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32_le<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64_le<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_f32_le<R: Read>(r: &mut R) -> Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
fn read_f64_le<R: Read>(r: &mut R) -> Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}
fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64_le(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| anyhow!("invalid utf8: {}", e))
}

fn read_value<R: Read>(r: &mut R, type_id: u32) -> Result<GgufValue> {
    match type_id {
        0 => Ok(GgufValue::U8(read_u8(r)?)),
        1 => Ok(GgufValue::I8(read_u8(r)? as i8)),
        2 => Ok(GgufValue::U16(read_u16_le(r)?)),
        3 => Ok(GgufValue::I16(read_u16_le(r)? as i16)),
        4 => Ok(GgufValue::U32(read_u32_le(r)?)),
        5 => Ok(GgufValue::I32(read_u32_le(r)? as i32)),
        6 => Ok(GgufValue::F32(read_f32_le(r)?)),
        7 => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        8 => Ok(GgufValue::String(read_string(r)?)),
        9 => {
            // GGUF array encoding: element type (u32) FIRST, then length (u64).
            let elem_type = read_u32_le(r)?;
            let n = read_u64_le(r)? as usize;
            let cap = n.min(10_000);
            let mut arr = Vec::with_capacity(cap);
            for _ in 0..n {
                arr.push(read_value(r, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(read_u64_le(r)?)),
        11 => Ok(GgufValue::I64(read_u64_le(r)? as i64)),
        12 => Ok(GgufValue::F64(read_f64_le(r)?)),
        _ => Err(anyhow!("unsupported gguf value type: {}", type_id)),
    }
}

pub fn parse_gguf_header(path: &Path) -> Result<GgufHeader> {
    // Buffered: GGUF headers carry large tokenizer arrays; reading them through an
    // unbuffered File is a syscall per primitive and makes header reads take seconds.
    let f = std::fs::File::open(path)?;
    let mut file = std::io::BufReader::with_capacity(1 << 20, f);
    let magic = read_u32_le(&mut file)?;
    if magic != GGUF_MAGIC {
        return Err(anyhow!("not a GGUF file (magic: 0x{:08X})", magic));
    }
    let version = read_u32_le(&mut file)?;
    let tensor_count = read_u64_le(&mut file)?;
    let metadata_kv_count = read_u64_le(&mut file)?;

    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_string(&mut file)?;
        let type_id = read_u32_le(&mut file)?;
        let value = read_value(&mut file, type_id)?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut file)?;
        let n_dims = read_u32_le(&mut file)?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64_le(&mut file)?);
        }
        let ggml_type = read_u32_le(&mut file)?;
        let offset = read_u64_le(&mut file)?;
        tensors.push(TensorInfo { name, n_dims, dims, ggml_type, offset });
    }

    Ok(GgufHeader { version, tensor_count, metadata_kv_count, metadata, tensors })
}

pub fn tensor_byte_size(t: &TensorInfo) -> u64 {
    let count: u64 = if t.dims.is_empty() { 1 } else { t.dims.iter().product() };
    let (block_ne, type_size) = ggml_type_layout(t.ggml_type);
    // Quantized ggml tensors are stored as blocks of `block_ne` elements packed into
    // `type_size` bytes; f16/f32/etc are block_ne = 1. Round up partial trailing blocks.
    let blocks = count.div_ceil(block_ne.max(1));
    blocks * type_size
}

/// (block size in elements, bytes per block) for each ggml tensor type.
/// Values match llama.cpp's `ggml_type_traits` (blck_size / type_size).
fn ggml_type_layout(gt: u32) -> (u64, u64) {
    match gt {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        _ => (1, 4),      // conservative fallback: assume 4 bytes/element
    }
}

pub fn sum_tensor_bytes(h: &GgufHeader) -> u64 {
    h.tensors.iter().map(tensor_byte_size).sum()
}

pub fn sum_block_tensor_bytes(h: &GgufHeader) -> (u64, u64) {
    let mut block_total: u64 = 0;
    let mut other_total: u64 = 0;
    for t in &h.tensors {
        let sz = tensor_byte_size(t);
        if is_block_tensor(&t.name) {
            block_total += sz;
        } else {
            other_total += sz;
        }
    }
    (block_total, other_total)
}

pub fn is_block_tensor(name: &str) -> bool {
    if name.contains("blk.") {
        return true;
    }
    let lower = name.to_lowercase();
    for marker in &["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
        if lower.contains(marker) {
            return true;
        }
    }
    false
}

pub fn arch_from_header(h: &GgufHeader) -> Option<String> {
    h.metadata.get("general.architecture").and_then(|v| v.as_str().map(|s| s.to_string()))
}

pub fn attention_type(h: &GgufHeader) -> Option<String> {
    h.metadata.get("general.attention.type")
        .or_else(|| h.metadata.get("attention.type"))
        .and_then(|v| v.as_str().map(|s| s.to_string()))
}

pub fn supported_standard_attention_archs() -> &'static [&'static str] {
    &["llama", "mistral", "gemma", "gemma2", "qwen2", "phi3", "phi", "starcoder2", "command-r", "deepseek2"]
}

pub fn file_type_quant_label(h: &GgufHeader) -> String {
    h.metadata.get("general.file_type")
        .and_then(|v| v.as_u32())
        .map(|ft| match ft {
            0 => "F32".into(), 1 => "F16".into(), 2 => "Q4_0".into(),
            3 => "Q4_1".into(), 7 => "Q8_0".into(), 8 => "Q8_1".into(),
            10 => "Q2_K".into(), 11 => "Q3_K_S".into(), 12 => "Q3_K_M".into(),
            13 => "Q3_K_L".into(), 14 => "Q4_K_S".into(), 15 => "Q4_K_M".into(),
            16 => "Q5_K_S".into(), 17 => "Q5_K_M".into(), 18 => "Q6_K".into(),
            _ => format!("Q{}", ft),
        })
        .unwrap_or_else(|| "unknown".into())
}
