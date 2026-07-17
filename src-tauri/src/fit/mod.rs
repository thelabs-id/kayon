use crate::ipc::*;
use crate::probe;
use chrono::Utc;

/// Reserve beyond weights + KV, in two parts. **Both are measured against the shipped Vulkan
/// backend** (§3) — the previous values were inherited CUDA folklore and over-reserved ~1.3 GB,
/// i.e. 16% of an 8 GB card, which made Kayon report "won't fit" for models that fit comfortably.
///
/// What llama.cpp actually allocates, read from its own load-time report on an RTX 4060:
///
/// | model        | n_vocab | ctx 2048 | ctx 4096 | ctx 16384 |
/// |--------------|---------|----------|----------|-----------|
/// | smollm2-135m |  49,152 |  97.1 Mi |  97.1 Mi |   97.1 Mi |
/// | llama-3.2-1b | 128,256 | 254.5 Mi | 258.5 Mi |  254.5 Mi |
/// | llama-3.2-3b | 128,256 | 278.5 Mi | 256.5 Mi |  256.5 Mi |
///
/// Two things fall out. The compute buffer is **flat in context length** (it is not the KV cache),
/// and it tracks **vocabulary**, not parameter count — it is dominated by the output logits,
/// `n_ubatch(512) × n_vocab × 4 bytes`, which is why the 1B and 3B agree to within noise while the
/// 135M (smaller vocab) is a third of the size. A 70B llama needs the same ~256 MiB as the 3B.
///
/// So the compute buffer is computed from the model's own vocabulary rather than assumed. A flat
/// constant is only safe until it isn't: 512 MiB would exactly equal Gemma 3's 262,144-token vocab
/// and under-reserve anything past it — a false "fits" on the largest-vocab models, which is the
/// dangerous direction.
const UBATCH: u64 = 512; // llama.cpp's default n_ubatch, which sizes the logits buffer
const LOGIT_BYTES: u64 = 4; // f32 logits
/// Floor for the graph itself, so a tiny-vocab model still reserves something sane.
const COMPUTE_BUFFER_FLOOR: u64 = 96 * 1024 * 1024;
/// Used only when the GGUF header didn't carry `vocab_size`. Covers a 393k vocab — half again the
/// largest shipping vocabulary — because guessing low here means an OOM the user was promised
/// wouldn't happen.
const COMPUTE_BUFFER_UNKNOWN_VOCAB: u64 = 768 * 1024 * 1024;

/// The compute buffer llama.cpp will allocate: `n_ubatch × n_vocab × 4`, floored.
///
/// Verified against llama.cpp's own report: smollm2-135m (vocab 49,152) → predicted 96 MiB,
/// observed 97.1; llama-3.2-1b/3b (vocab 128,256) → predicted 250.5 MiB, observed 254.5-256.5.
fn compute_buffer_bytes(vocab_size: Option<u32>) -> u64 {
    match vocab_size {
        Some(v) if v > 0 => std::cmp::max(UBATCH * v as u64 * LOGIT_BYTES, COMPUTE_BUFFER_FLOOR),
        _ => COMPUTE_BUFFER_UNKNOWN_VOCAB,
    }
}

/// Everything else the runtime holds on the device (graph nodes, staging, allocator slack). The
/// NVML-observed total for a loaded model came in *at or below* weights + KV + compute buffer, so
/// this is margin rather than a measured line item.
const RUNTIME_OVERHEAD: u64 = 128 * 1024 * 1024;
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
    value_length: Option<u32>,
    context_length: u32,
    kv_type_bytes: u8,
) -> u64 {
    // Key and value head dims can differ (some models set attention.key_length independently of
    // attention.value_length). Sum them rather than doubling one — otherwise KV is mis-estimated
    // for those models. Fall back to embedding_length/head_count only when a field is absent (§7).
    let default_dim = if head_count > 0 { embedding_length / head_count } else { 0 };
    let key_dim = key_length.unwrap_or(default_dim) as u64;
    // If only key_length is set, K and V head dims are symmetric — fall back to key_length before
    // the generic divisor (e.g. Gemma sets key_length=256 and V matches).
    let val_dim = value_length.or(key_length).unwrap_or(default_dim) as u64;
    let per_token =
        block_count as u64 * head_count_kv as u64 * (key_dim + val_dim) * kv_type_bytes as u64;
    per_token * context_length as u64
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
    let ram_total = sys.total_memory();

    evaluate_inner(
        model_id, quant_label, weight_bytes, None,
        arch, context_length, kv_type_bytes,
        vram_free, vram_total, ram_avail_raw, ram_total,
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
        vocab_size: gguf::vocab_size(&h),
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
    let ram_total = sys.total_memory();

    Ok(evaluate_inner(
        model_id, quant_label, w_total, per_block,
        &arch, context_length, kv_type_bytes,
        vram_free, vram_total, ram_avail_raw, ram_total,
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
    ram_total: u64,
) -> FitVerdict {
    // FIT-2: an explicit non-standard attention type ALWAYS wins over the architecture-name
    // default. A hybrid GGUF that happens to share an arch string with a standard model must
    // still return UNVERIFIED_ARCH — never a fabricated KV number.
    let att_type = arch.attention_type.as_deref().unwrap_or("standard");
    let explicit_nonstandard = matches!(att_type, "ssm" | "linear" | "hybrid" | "mamba" | "recurrent");
    let arch_supported = crate::gguf::supported_standard_attention_archs()
        .iter().any(|a| *a == arch.architecture);
    // Honest by default: an architecture we haven't validated the §7 KV model for returns
    // UNVERIFIED_ARCH rather than a fabricated number, even absent an explicit attention tag.
    let is_standard = !explicit_nonstandard && arch_supported;

    if !is_standard {
        // We can't model KV for this architecture, but we still avoid a fabricated all-layers
        // offload that would OOM a small GPU (RUN-1). Conservatively offload only as many whole
        // layers as the weights alone fit in VRAM (KV excluded because it's unknown); if we can't
        // estimate a per-layer size, fall back to CPU-only (0) rather than an all-layers offload.
        let vram_avail = vram_free.saturating_sub(display_headroom(vram_total));
        let per_block = per_block_override.unwrap_or(
            if arch.block_count > 0 { w_total / arch.block_count as u64 } else { 0 },
        );
        let budget = vram_avail.saturating_sub(compute_buffer_bytes(arch.vocab_size) + RUNTIME_OVERHEAD);
        let conservative_ngl = if per_block > 0 {
            ((budget / per_block) as i32).min(arch.block_count as i32).max(0)
        } else {
            0
        };
        return FitVerdict {
            model_id: model_id.to_string(),
            quant_label: quant_label.to_string(),
            context_length,
            kv_type_bytes,
            verdict: VerdictKind::UnverifiedArch,
            n_gpu_layers: conservative_ngl,
            per_block_bytes: if per_block > 0 { Some(per_block) } else { None },
            breakdown: None,
            explainability: format!(
                "Architecture '{}' with attention type '{}' is not in the supported-standard set. \
                 KV cache cannot be modeled, so the fit is unverified. Launching with a conservative \
                 {} GPU layers (weights-only estimate) to avoid an out-of-memory all-layers offload.",
                arch.architecture, att_type, conservative_ngl
            ),
            computed_at: Utc::now(),
        };
    }

    let kv = compute_kv_bytes(
        arch.block_count, arch.head_count_kv, arch.embedding_length,
        arch.head_count, arch.key_length, arch.value_length,
        context_length, kv_type_bytes,
    );

    let per_block = per_block_override.unwrap_or(
        if arch.block_count > 0 { w_total / arch.block_count as u64 } else { w_total },
    );
    let vram_avail = vram_free.saturating_sub(display_headroom(vram_total));
    // §7: OS_headroom = max(2 GB, 10% of *total* RAM) — reserved off the available pool.
    let ram_avail = ram_avail_raw.saturating_sub(os_headroom(ram_total));
    let comfort_margin = (vram_avail as f64 * COMFORT_MARGIN_RATIO) as u64;

    // Sized from this model's vocabulary, not a flat guess — see compute_buffer_bytes.
    let compute_buffer = compute_buffer_bytes(arch.vocab_size);
    let need_full = w_total + kv + compute_buffer + RUNTIME_OVERHEAD;

    let (verdict, ngl, breakdown_text) = if need_full <= vram_avail.saturating_sub(comfort_margin) {
        (VerdictKind::FitsFully, arch.block_count as i32, format!(
            "Weights {:.1} GB + KV {:.1} GB + buffers {:.1} GB = {:.1} GB vs {:.1} GB available (comfort margin {:.1} GB)",
            gb(w_total), gb(kv), gb(compute_buffer + RUNTIME_OVERHEAD), gb(need_full), gb(vram_avail), gb(comfort_margin)
        ))
    } else if need_full <= vram_avail {
        (VerdictKind::FitsTight, arch.block_count as i32, format!(
            "Weights {:.1} GB + KV {:.1} GB + buffers {:.1} GB = {:.1} GB fits within {:.1} GB but with thin headroom",
            gb(w_total), gb(kv), gb(compute_buffer + RUNTIME_OVERHEAD), gb(need_full), gb(vram_avail)
        ))
    } else {
        let gpu_budget = vram_avail.saturating_sub(compute_buffer + RUNTIME_OVERHEAD);
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
        // §7: the CPU-resident remainder is CPU-layer weights + KV for the CPU layers only. The
        // GPU already holds KV for its offloaded layers, so counting the full KV here would
        // double-count it against RAM and wrongly downgrade valid splits.
        let kv_off_gpu = kv.saturating_sub(kv_on_gpu_per_layer * ngl as u64);
        let cpu_remainder = w_total.saturating_sub(ngl as u64 * per_block) + kv_off_gpu;
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
        compute_buffer_bytes: compute_buffer,
        runtime_overhead_bytes: RUNTIME_OVERHEAD,
        vram_avail_bytes: vram_avail,
        ram_avail_bytes: ram_avail,
        headroom_display_bytes: display_headroom(vram_total),
        headroom_os_bytes: os_headroom(ram_total),
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
            // Llama-3 class vocabulary, so the fixtures reserve the ~255 MiB measured on real
            // hardware rather than the unknown-vocab fallback.
            vocab_size: Some(128_256),
            attention_type: None,
            runtime_min_version: None,
        }
    }

    #[test]
    fn fits_fully_on_a_big_gpu() {
        // 4 GB weights on a 24 GB card → comfortable fit.
        let v = evaluate_inner("m", "Q4", 4 * GIB, None, &llama_arch(), 4096, 2,
            24 * GIB, 24 * GIB, 32 * GIB, 32 * GIB);
        assert_eq!(v.verdict, VerdictKind::FitsFully);
        assert_eq!(v.n_gpu_layers, 32);
        assert!(v.breakdown.is_some());
    }

    #[test]
    fn exceeds_machine_when_bigger_than_both_pools() {
        // 200 GB weights, tiny machine → cannot fit anywhere.
        let v = evaluate_inner("m", "Q8", 200 * GIB, None, &llama_arch(), 4096, 2,
            8 * GIB, 8 * GIB, 8 * GIB, 8 * GIB);
        assert_eq!(v.verdict, VerdictKind::ExceedsMachine);
    }

    #[test]
    fn split_when_partially_offloadable() {
        // 20 GB weights on an 8 GB card with plenty of RAM → GPU/CPU split.
        let v = evaluate_inner("m", "Q6", 20 * GIB, None, &llama_arch(), 4096, 2,
            8 * GIB, 8 * GIB, 64 * GIB, 64 * GIB);
        assert_eq!(v.verdict, VerdictKind::GpuCpuSplit);
        assert!(v.n_gpu_layers > 0 && v.n_gpu_layers < 32);
    }

    #[test]
    fn cpu_only_when_nothing_fits_on_gpu_but_ram_holds_it() {
        // No usable VRAM, but the model fits in RAM.
        let v = evaluate_inner("m", "Q4", 6 * GIB, None, &llama_arch(), 4096, 2,
            0, 0, 64 * GIB, 64 * GIB);
        assert_eq!(v.verdict, VerdictKind::CpuOnly);
        assert_eq!(v.n_gpu_layers, 0);
    }

    #[test]
    fn unverified_arch_for_non_standard_attention() {
        let mut arch = llama_arch();
        arch.architecture = "mamba".into();
        arch.attention_type = Some("ssm".into());
        let v = evaluate_inner("m", "Q4", 4 * GIB, None, &arch, 4096, 2,
            24 * GIB, 24 * GIB, 32 * GIB, 32 * GIB);
        assert_eq!(v.verdict, VerdictKind::UnverifiedArch);
        assert!(v.breakdown.is_none());
    }

    #[test]
    fn explicit_hybrid_attention_overrides_standard_arch() {
        // Architecture string is standard ("llama") but attention is explicitly hybrid:
        // FIT-2 requires the explicit tag to win → UNVERIFIED_ARCH, not a fabricated KV number.
        let mut arch = llama_arch();
        arch.attention_type = Some("hybrid".into());
        let v = evaluate_inner("m", "Q4", 4 * GIB, None, &arch, 4096, 2,
            24 * GIB, 24 * GIB, 32 * GIB, 32 * GIB);
        assert_eq!(v.verdict, VerdictKind::UnverifiedArch);
    }

    #[test]
    fn unknown_arch_is_unverified() {
        // An architecture we haven't validated returns UNVERIFIED_ARCH rather than a guess.
        let mut arch = llama_arch();
        arch.architecture = "some-future-arch".into();
        let v = evaluate_inner("m", "Q4", 4 * GIB, None, &arch, 4096, 2,
            24 * GIB, 24 * GIB, 32 * GIB, 32 * GIB);
        assert_eq!(v.verdict, VerdictKind::UnverifiedArch);
    }

    #[test]
    fn q8_kv_knob_shrinks_kv() {
        let arch = llama_arch();
        let f16 = evaluate_inner("m", "Q4", 4 * GIB, None, &arch, 8192, 2, 24 * GIB, 24 * GIB, 32 * GIB, 64 * GIB);
        let q8 = evaluate_inner("m", "Q4", 4 * GIB, None, &arch, 8192, 1, 24 * GIB, 24 * GIB, 32 * GIB, 64 * GIB);
        let kv_f16 = f16.breakdown.unwrap().kv_bytes;
        let kv_q8 = q8.breakdown.unwrap().kv_bytes;
        assert_eq!(kv_f16, kv_q8 * 2, "q8_0 KV should be half of f16 KV");
    }

    #[test]
    fn compute_buffer_tracks_vocabulary_not_model_size() {
        // Pinned against llama.cpp's own load-time report on an RTX 4060 (Vulkan). The engine must
        // predict what the runtime will really allocate; these were measured, not derived.
        // Tolerance covers the small graph overhead on top of the logits buffer.
        let mib = |b: u64| b as f64 / 1048576.0;
        let within = |got: u64, want: f64| (mib(got) - want).abs() <= 12.0;

        // smollm2-135m, vocab 49,152 -> observed 97.1 MiB
        assert!(within(compute_buffer_bytes(Some(49_152)), 97.1),
            "got {} MiB", mib(compute_buffer_bytes(Some(49_152))));
        // llama-3.2-1b AND -3b share vocab 128,256 -> observed 254.5 and 256.5 MiB.
        // Same answer for both is the point: it does not scale with parameter count.
        assert!(within(compute_buffer_bytes(Some(128_256)), 255.0),
            "got {} MiB", mib(compute_buffer_bytes(Some(128_256))));

        // Gemma-class 262,144 vocab needs ~512 MiB — the case a flat 512 MiB constant would have
        // exactly exhausted, under-reserving into a false FITS.
        assert!(mib(compute_buffer_bytes(Some(262_144))) >= 512.0);

        // Unknown vocab must not guess low: reserve more than the largest shipping vocabulary.
        assert!(compute_buffer_bytes(None) > compute_buffer_bytes(Some(262_144)));
        assert!(compute_buffer_bytes(Some(0)) == compute_buffer_bytes(None), "0 is not a vocab");

        // A tiny vocab still reserves the graph floor rather than something absurd.
        assert_eq!(compute_buffer_bytes(Some(1_000)), COMPUTE_BUFFER_FLOOR);
    }
}
