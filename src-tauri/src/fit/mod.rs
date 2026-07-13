use crate::ipc::*;
use crate::probe;
use chrono::Utc;

const CUDA_OVERHEAD: u64 = 500 * 1024 * 1024;
const COMPUTE_BUFFER: u64 = 1024 * 1024 * 1024;
const COMFORT_MARGIN_RATIO: f64 = 0.10;

fn display_headroom(vram_total: u64) -> u64 {
    let pct = (vram_total as f64 * 0.10) as u64;
    std::cmp::max(1024 * 1024 * 1024, pct)
}

fn os_headroom(ram_total: u64) -> u64 {
    let pct = (ram_total as f64 * 0.10) as u64;
    std::cmp::max(2 * 1024 * 1024 * 1024, pct)
}

fn compute_kv_bytes(
    block_count: u32,
    head_count_kv: u32,
    embedding_length: u32,
    head_count: u32,
    key_length: Option<u32>,
    context_length: u32,
    kv_type_bytes: u8,
) -> u64 {
    let head_dim = key_length.unwrap_or_else(|| embedding_length / head_count);
    let d_kv = head_count_kv as u64 * head_dim as u64;
    let kv_per_token = 2u64 * block_count as u64 * d_kv * kv_type_bytes as u64;
    kv_per_token * context_length as u64
}

pub fn evaluate_remote(
    model_id: &str,
    quant_label: &str,
    weight_bytes: u64,
    arch: &ArchBlock,
    context_length: u32,
    kv_type_bytes: u8,
) -> FitVerdict {
    let vram_free = probe::get_vram_free();
    let vram_total = probe::get_vram_total();
    let sys = sysinfo::System::new_all();
    let ram_avail_raw = sys.available_memory();

    evaluate_inner(
        model_id, quant_label, weight_bytes, None,
        arch, context_length, kv_type_bytes,
        vram_free, vram_total, ram_avail_raw,
    )
}

/// Local (file present) verdict per FIT-1: parse the GGUF, sum the actual tensor bytes for an
/// exact `W_total`, derive the real per-block size, read the arch fields from the header, and
/// evaluate. This supersedes the remote (catalog `bytes`) approximation once the file lands.
pub fn evaluate_local(
    model_id: &str,
    quant_label: &str,
    path: &str,
    context_length: u32,
    kv_type_bytes: u8,
) -> anyhow::Result<FitVerdict> {
    use crate::gguf;
    let h = gguf::parse_gguf_header(std::path::Path::new(path))?;
    let architecture = gguf::arch_from_header(&h)
        .ok_or_else(|| anyhow::anyhow!("no general.architecture in GGUF"))?;
    let get = |k: &str| h.metadata.get(&format!("{architecture}.{k}")).and_then(|v| v.as_u32());

    let arch = ArchBlock {
        block_count: get("block_count").unwrap_or(0),
        head_count: get("attention.head_count").unwrap_or(1),
        head_count_kv: get("attention.head_count_kv").or_else(|| get("attention.head_count")).unwrap_or(1),
        embedding_length: get("embedding_length").unwrap_or(0),
        context_length: get("context_length").unwrap_or(context_length),
        key_length: get("attention.key_length"),
        value_length: get("attention.value_length"),
        attention_type: gguf::attention_type(&h),
        runtime_min_version: None,
        architecture: architecture.clone(),
    };

    let w_total = gguf::sum_tensor_bytes(&h);
    let (block_total, _other) = gguf::sum_block_tensor_bytes(&h);
    let per_block = if arch.block_count > 0 { Some(block_total / arch.block_count as u64) } else { None };

    let vram_free = probe::get_vram_free();
    let vram_total = probe::get_vram_total();
    let sys = sysinfo::System::new_all();
    let ram_avail_raw = sys.available_memory();

    Ok(evaluate_inner(
        model_id, quant_label, w_total, per_block,
        &arch, context_length, kv_type_bytes,
        vram_free, vram_total, ram_avail_raw,
    ))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_inner(
    model_id: &str,
    quant_label: &str,
    w_total: u64,
    per_block_override: Option<u64>,
    arch: &ArchBlock,
    context_length: u32,
    kv_type_bytes: u8,
    vram_free: u64,
    vram_total: u64,
    ram_avail_raw: u64,
) -> FitVerdict {
    let att_type = arch.attention_type.as_deref().unwrap_or("standard");
    let is_standard = att_type == "standard"
        || crate::gguf::supported_standard_attention_archs().iter().any(|a| *a == arch.architecture);

    if !is_standard {
        return FitVerdict {
            model_id: model_id.to_string(),
            quant_label: quant_label.to_string(),
            context_length,
            kv_type_bytes,
            verdict: VerdictKind::UnverifiedArch,
            n_gpu_layers: 999,
            per_block_bytes: None,
            breakdown: None,
            explainability: format!(
                "Architecture '{}' with attention type '{}' is not in the supported-standard set. \
                 KV cache cannot be modeled. The model may still load with auto-offload.",
                arch.architecture, att_type
            ),
            computed_at: Utc::now(),
        };
    }

    let kv = compute_kv_bytes(
        arch.block_count, arch.head_count_kv, arch.embedding_length,
        arch.head_count, arch.key_length.or(arch.value_length),
        context_length, kv_type_bytes,
    );

    let per_block = per_block_override.unwrap_or(
        if arch.block_count > 0 { w_total / arch.block_count as u64 } else { w_total },
    );
    let vram_avail = vram_free.saturating_sub(display_headroom(vram_total));
    let ram_avail = ram_avail_raw.saturating_sub(os_headroom(ram_avail_raw));
    let comfort_margin = (vram_avail as f64 * COMFORT_MARGIN_RATIO) as u64;

    let need_full = w_total + kv + COMPUTE_BUFFER + CUDA_OVERHEAD;

    let (verdict, ngl, breakdown_text) = if need_full <= vram_avail.saturating_sub(comfort_margin) {
        (VerdictKind::FitsFully, arch.block_count as i32, format!(
            "Weights {:.1} GB + KV {:.1} GB + buffers {:.1} GB = {:.1} GB vs {:.1} GB available (comfort margin {:.1} GB)",
            gb(w_total), gb(kv), gb(COMPUTE_BUFFER + CUDA_OVERHEAD), gb(need_full), gb(vram_avail), gb(comfort_margin)
        ))
    } else if need_full <= vram_avail {
        (VerdictKind::FitsTight, arch.block_count as i32, format!(
            "Weights {:.1} GB + KV {:.1} GB + buffers {:.1} GB = {:.1} GB fits within {:.1} GB but with thin headroom",
            gb(w_total), gb(kv), gb(COMPUTE_BUFFER + CUDA_OVERHEAD), gb(need_full), gb(vram_avail)
        ))
    } else {
        let gpu_budget = vram_avail.saturating_sub(COMPUTE_BUFFER + CUDA_OVERHEAD);
        let kv_on_gpu_per_layer = if arch.block_count > 0 { kv / arch.block_count as u64 } else { 0 };
        let mut ngl: i32 = 0;
        let mut gpu_used = 0u64;
        for i in 0..arch.block_count as i32 {
            let test = gpu_used + per_block + kv_on_gpu_per_layer;
            if test <= gpu_budget {
                gpu_used = test;
                ngl = i + 1;
            } else {
                break;
            }
        }

        let _ = gpu_used; // gpu_used already accounts for weights + on-GPU KV per layer
        let cpu_remainder = w_total.saturating_sub(ngl as u64 * per_block) + kv;
        if ngl > 0 && cpu_remainder <= ram_avail {
            (VerdictKind::GpuCpuSplit, ngl, format!(
                "GPU: {} of {} layers ({:.1} GB) + CPU: {:.1} GB | VRAM avail {:.1} GB, RAM avail {:.1} GB",
                ngl, arch.block_count, gb(ngl as u64 * per_block), gb(cpu_remainder), gb(vram_avail), gb(ram_avail)
            ))
        } else if ngl == 0 && w_total + kv <= ram_avail {
            (VerdictKind::CpuOnly, 0, format!(
                "No GPU layers fit. CPU-only: Weights {:.1} GB + KV {:.1} GB vs {:.1} GB RAM available",
                gb(w_total), gb(kv), gb(ram_avail)
            ))
        } else {
            (VerdictKind::ExceedsMachine, 0, format!(
                "Total need {:.1} GB exceeds VRAM ({:.1} GB) + RAM ({:.1} GB) combined",
                gb(w_total + kv), gb(vram_avail), gb(ram_avail)
            ))
        }
    };

    let breakdown = Some(VerdictBreakdown {
        weights_bytes: w_total,
        kv_bytes: kv,
        compute_buffer_bytes: COMPUTE_BUFFER,
        cuda_overhead_bytes: CUDA_OVERHEAD,
        vram_avail_bytes: vram_avail,
        ram_avail_bytes: ram_avail,
        headroom_display_bytes: display_headroom(vram_total),
        headroom_os_bytes: os_headroom(ram_avail_raw),
        comfort_margin_bytes: comfort_margin,
        total_need_bytes: need_full,
    });

    FitVerdict {
        model_id: model_id.to_string(),
        quant_label: quant_label.to_string(),
        context_length,
        kv_type_bytes,
        verdict,
        n_gpu_layers: ngl,
        per_block_bytes: Some(per_block),
        breakdown,
        explainability: breakdown_text,
        computed_at: Utc::now(),
    }
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn llama_arch() -> ArchBlock {
        ArchBlock {
            architecture: "llama".into(),
            block_count: 32,
            head_count: 32,
            head_count_kv: 8,
            embedding_length: 4096,
            context_length: 4096,
            key_length: None,
            value_length: None,
            attention_type: None,
            runtime_min_version: None,
        }
    }

    #[test]
    fn fits_fully_on_a_big_gpu() {
        // 4 GB weights on a 24 GB card → comfortable fit.
        let v = evaluate_inner("m", "Q4", 4 * GIB, None, &llama_arch(), 4096, 2,
            24 * GIB, 24 * GIB, 32 * GIB);
        assert_eq!(v.verdict, VerdictKind::FitsFully);
        assert_eq!(v.n_gpu_layers, 32);
        assert!(v.breakdown.is_some());
    }

    #[test]
    fn exceeds_machine_when_bigger_than_both_pools() {
        // 200 GB weights, tiny machine → cannot fit anywhere.
        let v = evaluate_inner("m", "Q8", 200 * GIB, None, &llama_arch(), 4096, 2,
            8 * GIB, 8 * GIB, 8 * GIB);
        assert_eq!(v.verdict, VerdictKind::ExceedsMachine);
    }

    #[test]
    fn split_when_partially_offloadable() {
        // 20 GB weights on an 8 GB card with plenty of RAM → GPU/CPU split.
        let v = evaluate_inner("m", "Q6", 20 * GIB, None, &llama_arch(), 4096, 2,
            8 * GIB, 8 * GIB, 64 * GIB);
        assert_eq!(v.verdict, VerdictKind::GpuCpuSplit);
        assert!(v.n_gpu_layers > 0 && v.n_gpu_layers < 32);
    }

    #[test]
    fn cpu_only_when_nothing_fits_on_gpu_but_ram_holds_it() {
        // No usable VRAM, but the model fits in RAM.
        let v = evaluate_inner("m", "Q4", 6 * GIB, None, &llama_arch(), 4096, 2,
            0, 0, 64 * GIB);
        assert_eq!(v.verdict, VerdictKind::CpuOnly);
        assert_eq!(v.n_gpu_layers, 0);
    }

    #[test]
    fn unverified_arch_for_non_standard_attention() {
        let mut arch = llama_arch();
        arch.architecture = "mamba".into();
        arch.attention_type = Some("ssm".into());
        let v = evaluate_inner("m", "Q4", 4 * GIB, None, &arch, 4096, 2,
            24 * GIB, 24 * GIB, 32 * GIB);
        assert_eq!(v.verdict, VerdictKind::UnverifiedArch);
        assert!(v.breakdown.is_none());
    }

    #[test]
    fn q8_kv_knob_shrinks_kv() {
        let arch = llama_arch();
        let f16 = evaluate_inner("m", "Q4", 4 * GIB, None, &arch, 8192, 2, 24 * GIB, 24 * GIB, 32 * GIB);
        let q8 = evaluate_inner("m", "Q4", 4 * GIB, None, &arch, 8192, 1, 24 * GIB, 24 * GIB, 32 * GIB);
        let kv_f16 = f16.breakdown.unwrap().kv_bytes;
        let kv_q8 = q8.breakdown.unwrap().kv_bytes;
        assert_eq!(kv_f16, kv_q8 * 2, "q8_0 KV should be half of f16 KV");
    }
}
