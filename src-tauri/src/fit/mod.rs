use crate::ipc::*;
use crate::probe;
use chrono::Utc;

/// Reserve beyond weights + KV, in two parts. **Both are measured against the shipped Vulkan
/// backend** (§3) at the exact llama.cpp build in `runtime-pin.json` — never inherited from CUDA
/// folklore, and never carried across a runtime bump without re-measuring.
///
/// That last clause is not hypothetical. The vocabulary model these constants replace was itself
/// measured, was correct for the build it was measured on, and is wrong here: upstream moved the
/// output logits out of VRAM entirely. Re-measure on every pin bump.
///
/// What b10056 actually allocates, from llama.cpp's own `-v` report on an RTX 4060:
///
/// | model        | n_embd | n_ff   | n_ctx | observed  | this model |
/// |--------------|--------|--------|-------|-----------|------------|
/// | smollm2-135m |    576 |  1,536 |  4096 |  15.26 Mi |      15.26 |
/// | llama-3.2-1b |  2,048 |  8,192 |  2048 |  58.01 Mi |      58.01 |
/// | llama-3.2-1b |  2,048 |  8,192 | 16384 |  72.01 Mi |      72.01 |
/// | llama-3.2-3b |  3,072 |  8,192 |  4096 |  64.01 Mi |      64.01 |
/// | llama-3.1-8b |  4,096 | 14,336 |  4096 | 104.01 Mi |     104.01 |
/// | llama-3.1-8b |  4,096 | 14,336 | 16384 | 116.01 Mi |     116.01 |
///
/// Two terms, both load-bearing:
///
/// - **Activations** — `n_ubatch × (2·n_embd + 3·n_ff) × 4`. Tracks the model's *width*; it is
///   independent of depth, parameter count, and vocabulary.
/// - **The f16 KQ mask** — `n_ubatch × n_ctx × 2`, i.e. 1 MiB per 1024 tokens of context. Identical
///   across every model measured, because it is head- and width-independent.
///
/// **The context term is the correctness fix.** The previous model had none: on the older build the
/// output logits (`n_ubatch × n_vocab × 4` ≈ 250 MiB) dominated the allocation and buried it.
/// b10056 moved the logits to a *host* buffer — `Vulkan_Host output buffer = n_seq × n_vocab × 4`,
/// 1.96 MiB of system RAM, not VRAM — so vocabulary now costs no VRAM at all and what remains grows
/// with context. A vocabulary-sized reserve is therefore both far too large at short contexts *and*
/// too small at long ones: smollm2 at ctx 131072 needs ~139 MiB where the old model returned its
/// 96 MiB floor. That is a false "fits", which is the direction that OOMs a user we promised.
///
/// `n_ctx` (not `n_ctx/n_parallel`) is the right context term for the flags RUN-1 actually passes:
/// llama-server defaults to `kv_unified = true`, which gives every slot the full context.
const UBATCH: u64 = 512; // llama.cpp's default n_ubatch, which sizes every activation in the graph
const ACT_BYTES: u64 = 4; // f32 activations
const MASK_BYTES: u64 = 2; // f16 KQ mask

/// Stand-in for `n_ff` when the arch block predates it (a catalog entry generated before the field
/// existed). The widest ratio in any shipping model is gemma-2-27b's 36,864/4,608 = 8x, so this
/// over-reserves for everything else — deliberately, because guessing low is the direction that OOMs.
const FF_RATIO_FALLBACK: u64 = 8;

/// Used when the arch block carries no width at all: nothing to compute from, so reserve enough to
/// cover a wide model rather than invent a number.
const COMPUTE_BUFFER_UNKNOWN: u64 = 512 * 1024 * 1024;

/// **Mixture-of-Experts is deliberately not modeled.** Its experts allocate activation the dense
/// width model does not describe, and it fails in the direction that OOMs: deepseek-coder-v2
/// (n_expert 64, n_expert_used 6, n_ff_exp 1408) allocated **154.88 MiB** where the dense model
/// predicts 76.12 — a 2x under-reserve, i.e. a false "fits". This matters for real users:
/// `deepseek2`, `mixtral`, `qwen2moe`, `qwen3moe`, `dbrx`, `granitemoe` and `olmoe` are all in the
/// supported-standard set and so get real verdicts, and MoE GGUFs arrive via Ollama adoption.
///
/// One measurement is not a model. Rather than fabricate an expert term from a single data point,
/// MoE reserves this flat figure *in place of the activation term* — comfortably above the one MoE
/// we have measured, and honestly labelled as conservative-because-unmeasured. Deriving the real
/// expert term needs several MoE models to measure against and is post-v1 (§7).
///
/// It replaces activations only; the KQ mask is still added on top. Taking `max()` of the two
/// instead would look equivalent and silently drop the expert reserve entirely once the mask grew
/// past it — at ctx 1M the mask alone is 1 GiB, and MoE would have reserved *nothing* for experts.
const COMPUTE_BUFFER_MOE_UNMEASURED: u64 = 512 * 1024 * 1024;

/// The VRAM compute buffer llama.cpp will allocate: activations (model width) + KQ mask (context).
///
/// Predicts llama.cpp's own reported figure to within 0.01 MiB across 135M→8B, two architecture
/// families, three vocabularies, and ctx 1024→16384 — see the table above. Vocabulary is
/// deliberately absent: as of b10056 the vocabulary-sized buffer lives in host RAM.
fn compute_buffer_bytes(arch: &ArchBlock, context_length: u32) -> u64 {
    let n_embd = arch.embedding_length as u64;
    if n_embd == 0 {
        return COMPUTE_BUFFER_UNKNOWN;
    }
    let mask = UBATCH * context_length as u64 * MASK_BYTES;

    // Experts allocate what the dense model can't see. Reserve rather than under-predict. The mask
    // is context-driven and applies to MoE the same as anything else, so it adds rather than maxes.
    if arch.expert_count.is_some_and(|n| n > 0) {
        return COMPUTE_BUFFER_MOE_UNMEASURED.saturating_add(mask);
    }

    let n_ff = arch
        .feed_forward_length
        .map(u64::from)
        .unwrap_or(n_embd * FF_RATIO_FALLBACK);
    let activations = UBATCH * (2 * n_embd + 3 * n_ff) * ACT_BYTES;
    activations + mask
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
        feed_forward_length: get("feed_forward_length"),
        expert_count: get("expert_count"),
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
        // Not `compute_buffer_bytes` — the width model above was measured on standard-attention
        // dense models, and this branch exists precisely because this architecture is not one.
        // Applying its activation term here would be the same fabrication the verdict refuses to
        // make about KV. But we still don't want to *under*-reserve: reserve the unknown-width
        // figure PLUS the context-scaled term, so the budget shrinks monotonically with context and
        // never sits below what the dense model would have taken (whose activations are always
        // below the 512 MiB floor for any real arch). Same shape as the MoE reserve, same reason.
        let unverified_reserve =
            COMPUTE_BUFFER_UNKNOWN.saturating_add(UBATCH * context_length as u64 * MASK_BYTES);
        let budget = vram_avail.saturating_sub(unverified_reserve + RUNTIME_OVERHEAD);
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

    // Sized from this model's width and the requested context, not a flat guess — see
    // compute_buffer_bytes. It grows with `context_length`, so it must be computed after it.
    let compute_buffer = compute_buffer_bytes(arch, context_length);
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
            // Llama-3.1-8b's real width, so the fixtures reserve the 104 MiB measured on real
            // hardware rather than the conservative missing-width fallback.
            feed_forward_length: Some(14_336),
            expert_count: None, // dense
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

    /// An arch block with a given width, for the compute-buffer measurements.
    fn arch_wh(embedding_length: u32, feed_forward_length: u32) -> ArchBlock {
        ArchBlock { embedding_length, feed_forward_length: Some(feed_forward_length), ..llama_arch() }
    }

    #[test]
    fn compute_buffer_matches_what_llamacpp_really_allocates() {
        // Every figure below is llama.cpp's own `-v` report on an RTX 4060 against the pinned
        // Vulkan build (runtime-pin.json, b10056). These were measured, not derived — the engine's
        // whole claim is that it predicts the runtime, so the test asserts against the runtime.
        let mib = |b: u64| b as f64 / 1048576.0;
        // Tight on purpose: the model reproduced every observation to 0.01 MiB. A loose tolerance
        // here would hide exactly the kind of drift that made the previous model wrong.
        let obs = |arch: &ArchBlock, ctx: u32, want: f64| {
            let got = mib(compute_buffer_bytes(arch, ctx));
            assert!((got - want).abs() <= 0.05, "ctx {ctx}: predicted {got} MiB, llama.cpp allocated {want} MiB");
        };

        obs(&arch_wh(576, 1_536), 4096, 15.26);      // smollm2-135m
        obs(&arch_wh(2_048, 8_192), 2048, 58.01);    // llama-3.2-1b
        obs(&arch_wh(2_048, 8_192), 16384, 72.01);   // llama-3.2-1b, long context
        obs(&arch_wh(3_072, 8_192), 4096, 64.01);    // llama-3.2-3b
        obs(&arch_wh(4_096, 14_336), 4096, 104.01);  // llama-3.1-8b
        obs(&arch_wh(4_096, 14_336), 16384, 116.01); // llama-3.1-8b, long context
    }

    #[test]
    fn compute_buffer_grows_with_context_and_ignores_vocabulary() {
        // The correctness fix. The previous model was flat in context because the old build's
        // logits buffer buried the context term; b10056 moved the logits to host RAM. A model with
        // no context term under-reserves at long context, which is a false FITS.
        let a = arch_wh(2_048, 8_192);
        let short = compute_buffer_bytes(&a, 2048);
        let long = compute_buffer_bytes(&a, 131_072);
        assert!(long > short, "compute buffer must grow with context");
        // 1 MiB per 1024 tokens of context, measured identically on every model.
        assert_eq!(long - short, UBATCH * (131_072 - 2048) * MASK_BYTES);

        // Vocabulary costs no VRAM as of b10056: the output-logits buffer is host-side.
        let mut big_vocab = arch_wh(2_048, 8_192);
        big_vocab.vocab_size = Some(262_144);
        assert_eq!(compute_buffer_bytes(&big_vocab, 4096), compute_buffer_bytes(&a, 4096));
    }

    #[test]
    fn moe_reserves_above_what_the_dense_model_would_have_guessed() {
        // deepseek-coder-v2 (n_embd 2048, n_ff 10944, n_expert 64, n_expert_used 6, n_ff_exp 1408)
        // allocated 154.88 MiB at ctx 4096; the dense model predicts 76.12. Shipping the dense
        // number for MoE would be a 2x under-reserve — a false FITS on seven supported arch
        // families. Until the expert term is actually measured, MoE reserves conservatively.
        let mut moe = arch_wh(2_048, 10_944);
        moe.architecture = "deepseek2".into();
        moe.expert_count = Some(64);

        let dense = compute_buffer_bytes(&arch_wh(2_048, 10_944), 4096);
        let got = compute_buffer_bytes(&moe, 4096);
        let mib = |b: u64| b as f64 / 1048576.0;

        assert!(mib(dense) < 154.88, "fixture guard: the dense model is the under-prediction");
        assert!(mib(got) >= 154.88, "MoE must cover the 154.88 MiB really allocated, got {}", mib(got));
        assert!(got > dense, "MoE must reserve above the dense estimate");

        // The expert reserve must survive a long context rather than be swallowed by the mask: a
        // `max(flat, mask)` would silently reserve *nothing* for experts once the mask grew past
        // the flat figure, which at ctx 1M is 1 GiB.
        let huge = compute_buffer_bytes(&moe, 1_048_576);
        let mask_1m = UBATCH * 1_048_576 * MASK_BYTES;
        assert!(huge > mask_1m, "the expert reserve must not vanish under a large mask");
        assert_eq!(huge - mask_1m, COMPUTE_BUFFER_MOE_UNMEASURED);
    }

    #[test]
    fn unverified_arch_does_not_borrow_the_dense_width_model() {
        // The budget for an unmodellable architecture must not be computed with a buffer model
        // measured on dense standard-attention models — that is the fabrication the verdict itself
        // refuses to make. Widening the model must not change an unverified arch's offload.
        let mut narrow = llama_arch();
        narrow.architecture = "some-future-arch".into();
        narrow.embedding_length = 1_024;
        narrow.feed_forward_length = Some(2_048);

        let mut wide = narrow.clone();
        wide.embedding_length = 8_192;
        wide.feed_forward_length = Some(28_672);

        let a = evaluate_inner("m", "Q4", 8 * GIB, Some(GIB / 4), &narrow, 4096, 2, 24 * GIB, 24 * GIB, 32 * GIB, 32 * GIB);
        let b = evaluate_inner("m", "Q4", 8 * GIB, Some(GIB / 4), &wide, 4096, 2, 24 * GIB, 24 * GIB, 32 * GIB, 32 * GIB);
        assert_eq!(a.verdict, VerdictKind::UnverifiedArch);
        assert_eq!(b.verdict, VerdictKind::UnverifiedArch);
        assert_eq!(a.n_gpu_layers, b.n_gpu_layers, "unverified ngl must not depend on the dense width model");

        // Conservatism must be monotonic in context: a longer context can only offload the same or
        // fewer layers, never more. A flat reserve with no context term would let a huge context
        // offload as much as a small one, then OOM on the runtime buffers it ignored.
        let short = evaluate_inner("m", "Q4", 8 * GIB, Some(GIB / 4), &narrow, 4096, 2, 24 * GIB, 24 * GIB, 32 * GIB, 32 * GIB);
        let long = evaluate_inner("m", "Q4", 8 * GIB, Some(GIB / 4), &narrow, 131_072, 2, 24 * GIB, 24 * GIB, 32 * GIB, 32 * GIB);
        assert!(long.n_gpu_layers <= short.n_gpu_layers, "a longer context must not offload MORE layers");
    }

    #[test]
    fn compute_buffer_never_guesses_low_when_width_is_missing() {
        // A catalog entry generated before feed_forward_length existed: fall back to the widest
        // shipping ratio (gemma-2-27b, 8x) rather than under-reserve.
        let mut no_ff = arch_wh(4_096, 14_336);
        no_ff.feed_forward_length = None;
        assert!(compute_buffer_bytes(&no_ff, 4096) > compute_buffer_bytes(&arch_wh(4_096, 14_336), 4096));
        assert_eq!(
            compute_buffer_bytes(&no_ff, 4096),
            compute_buffer_bytes(&arch_wh(4_096, 4_096 * FF_RATIO_FALLBACK as u32), 4096)
        );

        // No width at all — nothing to compute from, so reserve rather than invent.
        let mut nothing = arch_wh(4_096, 14_336);
        nothing.embedding_length = 0;
        assert_eq!(compute_buffer_bytes(&nothing, 4096), COMPUTE_BUFFER_UNKNOWN);
    }
}
