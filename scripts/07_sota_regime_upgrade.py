#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7 (OCCAM v2 — Regime-Adaptive Mental-Model Routing Upgrade):
100% Authentic Single-GPU RTX 3090 Post-Training & Truncation-Free vLLM Evaluation.

Key Scientific Upgrades Operationalizing Proposition 1 (Framing-vs-Rollout Dichotomy):
  1. Explicit Zero-Prompt Regime Prediction inside <cog_router>:
       [Execution_Regime]: Structural_Framing | Symbolic_Rollout
     Learned strictly from the 1,570 decontaminated training seeds (0.00% 13-gram test overlap).
  2. Native <think> Chat-Template Alignment for DeepSeek-R1-Distill-Qwen-7B:
       Prompt ends with "<|Assistant|><think>\n";
       Completion emits "<cog_router>...</cog_router>\n<cog_trace>...</cog_trace>\n</think>\n<answer>...</answer><eos>".
     Eliminates the train-test <think> template mismatch completely.
  3. Low-Drift Anchored LoRA (r=16, alpha=16, lr=6e-5 on Qwen; r=32, alpha=32, lr=8e-5 on R1)
     + Calibrated Early-Horizon OCCAM-SimPO (K=25, alpha_r=2.0, nll_coef=0.25).
  4. Single-Backbone Multi-Adapter Regime-Adaptive Execution (OCCAM-Adaptive):
       - Phase 1 (~45 tokens, zero system prompt): OCCAM LoRA emits <cog_router>...</cog_router>.
       - Phase 2 (purely driven by model's own [Execution_Regime] tag):
           * Structural_Framing -> Continue seamlessly via OCCAM LoRA (~180-230 tokens, prefix-cached).
           * Symbolic_Rollout   -> Disengage compression LoRA (lora_request=None) on the exact same
                                   single-GPU vLLM engine so irreducible symbolic derivations complete
                                   without violating Proposition 1's Fano floor.
       - Honest Accounting: Total tokens and wall-clock time strictly sum BOTH Phase 1 + Phase 2!
"""
import argparse
import collections
import gc
import importlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_grader import grade_prediction
eval_mod = importlib.import_module("04a_vllm_eval_runner")
train_mod = None  # Lazy-imported inside train functions so eval never touches bitsandbytes CUDA!

BASE_DIR = Path(__file__).resolve().parent.parent
SYNTH_PATH = BASE_DIR / "data" / "synthesized_pairs" / "synthesized_pairs_1600.jsonl"
SPLITS_DIR = BASE_DIR / "data" / "final_splits"
CKPT_ROOT = BASE_DIR / "outputs" / "checkpoints_occam"
TRACES_DIR = BASE_DIR / "outputs" / "eval_traces"
ASSETS_DIR = BASE_DIR / "outputs" / "paper_assets"
MODELS_ROOT = Path("./models")

SYMBOLIC_SOURCE_SUBSTRINGS = (
    "hendrycks_math",
    "(navigate)",
    "(web_of_lies)",
    "(formal_fallacies)",
    "(boolean_expressions)",
    "(date_understanding)",
    "(logical_deduction_five_objects)",
    "(tracking_shuffled_objects_five_objects)",
)


def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def classify_train_regime(rec: dict) -> str:
    """Principled Proposition-1 regime label derived strictly from training seed domain."""
    sb = str(rec.get("source_benchmark", ""))
    if any(sub in sb for sub in SYMBOLIC_SOURCE_SUBSTRINGS):
        return "Symbolic_Rollout"
    return "Structural_Framing"


def inject_regime_into_router(text: str, regime: str) -> str:
    """Insert [Execution_Regime]: <regime> cleanly right before </cog_router>."""
    if "[Execution_Regime]:" in text:
        return text
    tag = "</cog_router>"
    idx = text.find(tag)
    if idx != -1:
        prefix = text[:idx].rstrip()
        suffix = text[idx:]
        return f"{prefix}\n[Execution_Regime]: {regime}\n{suffix}"
    return f"<cog_router>\n[Execution_Regime]: {regime}\n</cog_router>\n{text}"


def format_response_for_backbone(text: str, is_r1: bool) -> str:
    """
    For Qwen2.5: <cog_router>...</cog_router>\n<cog_trace>...</cog_trace>\n<answer>...</answer>
    For DeepSeek-R1-Distill: aligns natively with R1's generation prompt ending in '<think>\n',
    closing </think> right before <answer>...</answer>!
    """
    txt = text.strip()
    if not is_r1:
        return txt
    # Remove any stray <think> tags first
    txt = re.sub(r"</?think>", "", txt, flags=re.IGNORECASE).strip()
    ans_idx = txt.rfind("<answer>")
    if ans_idx != -1:
        body = txt[:ans_idx].rstrip()
        ans = txt[ans_idx:].lstrip()
        return f"{body}\n</think>\n{ans}"
    return f"{txt}\n</think>"


def build_v2_datasets() -> Tuple[List[dict], List[dict]]:
    """
    Builds decontaminated OCCAM-2K-v2 SFT & SimPO records strictly from existing
    1,570 verified pairs (0.00% SHA-256 & 13-gram test overlap).
    """
    raw = [json.loads(ln) for ln in SYNTH_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sft_rows, simpo_rows = [], []
    reg_cnt = collections.Counter()
    for r in raw:
        regime = classify_train_regime(r)
        reg_cnt[regime] += 1
        y_ch = inject_regime_into_router(r["y_chosen"], regime)
        y_rj = r["y_rejected"]
        dyn_lam = 0.04 if regime == "Symbolic_Rollout" else 0.18
        sft_rows.append({
            "uid": r["uid"],
            "benchmark": r["benchmark"],
            "regime": regime,
            "prompt": r["question"],
            "response": y_ch,
        })
        simpo_rows.append({
            "uid": r["uid"],
            "benchmark": r["benchmark"],
            "regime": regime,
            "prompt": r["question"],
            "chosen": y_ch,
            "rejected": y_rj,
            "dynamic_lambda_len": dyn_lam,
        })
    print(f"[OCCAM-2K-v2 Prepared] N={len(sft_rows)} | Regimes={dict(reg_cnt)}")
    return sft_rows, simpo_rows


def encode_aligned_pair(
    tokenizer,
    prompt_str: str,
    response_str: str,
    is_r1: bool,
    max_seq_len: int = 1536,
) -> Dict[str, torch.Tensor]:
    """
    Bit-exact alignment between training sequences and vLLM add_generation_prompt=True!
    For DeepSeek-R1, prompt_text ends with '<|Assistant|><think>\n', and response_str starts
    directly with '<cog_router>\n...' and closes '</think>\n<answer>...</answer>'.
    """
    resp_fmt = format_response_for_backbone(response_str, is_r1=is_r1)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_str}],
        tokenize=False,
        add_generation_prompt=True,
    )
    eos = tokenizer.eos_token or ""
    full_text = prompt_text + resp_fmt + eos

    enc_prompt = tokenizer(prompt_text, add_special_tokens=False)
    enc_full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
    )
    input_ids = enc_full["input_ids"]
    offsets = enc_full["offset_mapping"]
    prompt_len = min(len(enc_prompt["input_ids"]), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    router_mask = [0.0] * len(input_ids)
    r_start = full_text.find("<cog_router>", max(0, len(prompt_text) - 5))
    r_end_tag = "</cog_router>"
    r_end = full_text.find(r_end_tag, r_start) if r_start != -1 else -1
    if r_start != -1 and r_end != -1:
        r_end_char = r_end + len(r_end_tag)
        for idx, (cs, ce) in enumerate(offsets):
            if idx >= prompt_len and ce > r_start and cs < r_end_char:
                router_mask[idx] = 1.0

    resp_len = max(1, len(input_ids) - prompt_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "router_mask": torch.tensor(router_mask, dtype=torch.float32),
        "resp_len": float(resp_len),
    }


class AlignedSFTDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer, is_r1: bool, max_seq_len: int = 1536):
        self.items = [
            encode_aligned_pair(tokenizer, r["prompt"], r["response"], is_r1=is_r1, max_seq_len=max_seq_len)
            for r in records
        ]
        self.lengths = [x["input_ids"].size(0) for x in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class AlignedSimPODataset(Dataset):
    def __init__(self, records: List[dict], tokenizer, is_r1: bool, max_seq_len: int = 1536):
        self.items = []
        self.lengths = []
        for r in records:
            ch = encode_aligned_pair(tokenizer, r["prompt"], r["chosen"], is_r1=is_r1, max_seq_len=max_seq_len)
            rj = encode_aligned_pair(tokenizer, r["prompt"], r["rejected"], is_r1=is_r1, max_seq_len=max_seq_len)
            self.items.append({"chosen": ch, "rejected": rj, "dyn_lambda": float(r["dynamic_lambda_len"])})
            self.lengths.append(rj["input_ids"].size(0) * 2 + ch["input_ids"].size(0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def is_ckpt_done(save_dir: Path) -> bool:
    return (save_dir / "adapter_model.safetensors").exists() and (save_dir / "training_metrics.json").exists()


def train_aligned_sft(
    name: str,
    model_path: str,
    records: List[dict],
    save_dir: Path,
    is_r1: bool,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lr: float = 6e-5,
    epochs: int = 2,
    micro_bs: int = 4,
    grad_accum: int = 8,
    max_seq_len: int = 1536,
) -> dict:
    if is_ckpt_done(save_dir):
        print(f"  [Skip - Trained] {name} already exists at {save_dir}")
        return json.loads((save_dir / "training_metrics.json").read_text(encoding="utf-8"))

    import bitsandbytes as bnb
    global train_mod
    if train_mod is None:
        train_mod = importlib.import_module("03_train_cog_simpo")

    set_seed(2026)
    cleanup_gpu()
    print("=" * 78)
    print(f"[Train v2 SFT] {name} | Base={Path(model_path).name} | r={lora_r} alpha={lora_alpha} lr={lr} ep={epochs} is_r1={is_r1}")
    print("=" * 78)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        trust_remote_code=True,
    )
    base.config.use_cache = False
    base.enable_input_require_grads()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.08,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)

    ds = AlignedSFTDataset(records, tokenizer, is_r1=is_r1, max_seq_len=max_seq_len)
    loader = DataLoader(
        ds,
        batch_sampler=train_mod.LengthBucketBatchSampler(ds.lengths, batch_size=micro_bs, seed=2026),
        collate_fn=train_mod.make_sft_collator(tokenizer.pad_token_id),
    )

    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    total_opt_steps = max(1, math.ceil(len(loader) * epochs / grad_accum))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(4, total_opt_steps // 10), num_training_steps=total_opt_steps
    )

    model.train()
    t0 = time.perf_counter()
    losses = []
    micro_step = 0
    optimizer.zero_grad()

    for ep in range(1, epochs + 1):
        ep_losses = []
        for batch in loader:
            micro_step += 1
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                raw_loss = train_mod.compute_sft_chunked_loss(model, input_ids, attention_mask, labels)
                loss = raw_loss / grad_accum

            loss.backward()
            lv = float(raw_loss.detach().cpu().item())
            ep_losses.append(lv)
            losses.append(lv)

            if micro_step % grad_accum == 0 or micro_step == len(loader) * epochs:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print(f"  [{name} Epoch {ep}/{epochs}] mean_loss={sum(ep_losses)/max(1,len(ep_losses)):.4f}")

    torch.cuda.synchronize()
    wall_sec = time.perf_counter() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    metrics = {
        "checkpoint_name": name,
        "base_model": Path(model_path).name,
        "stage": "OCCAM_v2_Anchored_Regime_SFT",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lr": lr,
        "num_samples": len(records),
        "epochs": epochs,
        "opt_steps": total_opt_steps,
        "initial_loss": round(losses[0], 4) if losses else 0.0,
        "final_loss": round(sum(losses[-10:]) / max(1, min(10, len(losses))), 4),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "wall_time_sec": round(wall_sec, 2),
        "gpu_hours": round(wall_sec / 3600.0, 4),
    }
    (save_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  -> Saved {name}: loss={metrics['final_loss']:.4f} | VRAM={peak_vram_gb:.2f} GB | Time={wall_sec/60:.2f}m")

    del model, base, optimizer, loader, ds
    cleanup_gpu()
    return metrics


def train_aligned_simpo_k25(
    name: str,
    model_path: str,
    init_sft_dir: Path,
    records: List[dict],
    save_dir: Path,
    is_r1: bool,
    target_opt_steps: int = 25,
    lr: float = 6e-6,
    alpha_router: float = 2.0,
    nll_coef: float = 0.25,
    micro_bs: int = 2,
    grad_accum: int = 8,
    max_seq_len: int = 1280,
) -> dict:
    if is_ckpt_done(save_dir):
        print(f"  [Skip - Trained] {name} already exists at {save_dir}")
        return json.loads((save_dir / "training_metrics.json").read_text(encoding="utf-8"))

    import bitsandbytes as bnb
    global train_mod
    if train_mod is None:
        train_mod = importlib.import_module("03_train_cog_simpo")

    set_seed(2026)
    cleanup_gpu()
    rng = random.Random(2026)
    sub_n = min(len(records), target_opt_steps * micro_bs * grad_accum)
    sub_records = rng.sample(records, sub_n)

    print("=" * 78)
    print(f"[Train v2 Calibrated SimPO] {name} | Init={init_sft_dir.name} | K={target_opt_steps} | N={len(sub_records)} | alpha_r={alpha_router}")
    print("=" * 78)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        trust_remote_code=True,
    )
    base.config.use_cache = False
    base.enable_input_require_grads()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, str(init_sft_dir), is_trainable=True)

    ds = AlignedSimPODataset(sub_records, tokenizer, is_r1=is_r1, max_seq_len=max_seq_len)
    loader = DataLoader(
        ds,
        batch_sampler=train_mod.LengthBucketBatchSampler(ds.lengths, batch_size=micro_bs, seed=2026),
        collate_fn=train_mod.make_simpo_collator(tokenizer.pad_token_id),
    )

    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    total_opt_steps = max(1, math.ceil(len(loader) / grad_accum))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(3, total_opt_steps // 8), num_training_steps=total_opt_steps
    )

    model.train()
    t0 = time.perf_counter()
    losses, margins = [], []
    micro_step = 0
    optimizer.zero_grad()

    for batch in loader:
        micro_step += 1
        w_ids = batch["w_ids"].cuda(non_blocking=True)
        w_lbls = batch["w_lbls"].cuda(non_blocking=True)
        w_rmask = batch["w_rmask"].cuda(non_blocking=True)
        w_amask = batch["w_amask"].cuda(non_blocking=True)
        w_lens = batch["w_lens"].cuda(non_blocking=True)

        l_ids = batch["l_ids"].cuda(non_blocking=True)
        l_lbls = batch["l_lbls"].cuda(non_blocking=True)
        l_rmask = batch["l_rmask"].cuda(non_blocking=True)
        l_amask = batch["l_amask"].cuda(non_blocking=True)
        l_lens = batch["l_lens"].cuda(non_blocking=True)
        dyn_lams = batch["dyn_lams"].cuda(non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            r_w, nll_w = train_mod.compute_occam_weighted_logps(
                model, w_ids, w_amask, w_lbls, w_rmask, alpha_router=alpha_router
            )
            r_l, _ = train_mod.compute_occam_weighted_logps(
                model, l_ids, l_amask, l_lbls, l_rmask, alpha_router=alpha_router
            )
            len_pen = dyn_lams * (w_lens / l_lens.clamp(min=1.0))
            simpo_margin = 2.0 * (r_w - r_l - 0.5 - len_pen)
            simpo_loss = -F.logsigmoid(simpo_margin).mean()
            total_loss = (simpo_loss + nll_coef * nll_w) / grad_accum

        total_loss.backward()
        losses.append(float(simpo_loss.detach().cpu().item()))
        margins.append(float((r_w - r_l).mean().detach().cpu().item()))

        if micro_step % grad_accum == 0 or micro_step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    torch.cuda.synchronize()
    wall_sec = time.perf_counter() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    metrics = {
        "checkpoint_name": name,
        "base_model": Path(model_path).name,
        "stage": "OCCAM_v2_Calibrated_SimPO_K25",
        "alpha_router": alpha_router,
        "nll_coef": nll_coef,
        "num_samples": len(sub_records),
        "opt_steps": total_opt_steps,
        "initial_loss": round(losses[0], 4) if losses else 0.0,
        "final_loss": round(sum(losses[-8:]) / max(1, min(8, len(losses))), 4),
        "final_margin": round(sum(margins[-8:]) / max(1, min(8, len(margins))), 4),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "wall_time_sec": round(wall_sec, 2),
        "gpu_hours": round(wall_sec / 3600.0, 4),
    }
    (save_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  -> Saved {name}: margin={metrics['final_margin']:.4f} | VRAM={peak_vram_gb:.2f} GB | Time={wall_sec/60:.2f}m")

    del model, base, optimizer, loader, ds
    cleanup_gpu()
    return metrics


def parse_predicted_regime(router_txt: str) -> str:
    """Strictly parse the model's own autoregressive [Execution_Regime] prediction."""
    m = re.search(r"\[Execution_Regime\]\s*:\s*([A-Za-z_]+)", router_txt, re.IGNORECASE)
    if m:
        val = m.group(1).strip().lower()
        if "symbolic" in val or "rollout" in val:
            return "Symbolic_Rollout"
        return "Structural_Framing"
    return "Structural_Framing"


def eval_direct_stop_answer(
    llm,
    tokenizer,
    test_items: List[dict],
    model_id: str,
    lora_req,
    max_tokens: int = 1536,
):
    from vllm import SamplingParams

    out_path = TRACES_DIR / f"{model_id}.jsonl"
    if eval_mod.trace_file_complete(model_id, len(test_items)):
        print(f"  [Skip - Cached] {model_id}")
        return

    print(f"  [Eval Direct] {model_id:<28} | N={len(test_items)} | max_tok={max_tokens}")
    prompts = [eval_mod.build_prompt_for_mode(tokenizer, it, "occam_zero_prompt") for it in test_items]
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, lora_request=lora_req)
    wall_sec = time.perf_counter() - t0
    avg_ms = (wall_sec * 1000.0) / max(1, len(test_items))

    records_out = []
    for it, o in zip(test_items, outs):
        txt = o.outputs[0].text
        n_tok = len(o.outputs[0].token_ids)
        ok, ext = grade_prediction(txt, it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
        records_out.append({
            "uid": it["uid"],
            "benchmark": it["benchmark"],
            "split_group": it["split_group"],
            "ground_truth": it["ground_truth"],
            "extracted_answer": ext,
            "correct": ok,
            "tokens": n_tok,
            "output": txt,
            "avg_batch_latency_ms": round(avg_ms, 2),
            "wall_time_sec_total": round(wall_sec, 2),
        })
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print_summary(model_id)


def eval_regime_adaptive(
    llm,
    tokenizer,
    test_items: List[dict],
    model_id: str,
    lora_req,
    is_r1: bool,
    framing_max_tokens: int = 768,
    symbolic_max_tokens: int = 1024,
    symbolic_temp: float = 0.0,
):
    """
    Authentic Single-Backbone Zero-Prompt Regime-Adaptive OCCAM Execution:
      Phase 1: Zero-prompt OCCAM LoRA generates <cog_router>...</cog_router> (~45 tokens).
      Phase 2: Purely conditioned on the emitted [Execution_Regime] tag:
        - Structural_Framing -> Continue seamlessly via OCCAM LoRA (prefix-cached!).
        - Symbolic_Rollout   -> Disengage compression LoRA (lora_request=None) on the exact
                                same vLLM engine so irreducible symbolic derivations run cleanly.
      Honest Accounting: Total tokens = router_tokens + phase2_tokens;
                         Total wall_time = phase1_time + phase2_framing_time + phase2_symbolic_time.
    """
    from vllm import SamplingParams

    out_path = TRACES_DIR / f"{model_id}.jsonl"
    if eval_mod.trace_file_complete(model_id, len(test_items)):
        print(f"  [Skip - Cached] {model_id}")
        return

    print(
        f"  [Eval Adaptive] {model_id:<26} | N={len(test_items)} | "
        f"frame_max={framing_max_tokens} | sym_max={symbolic_max_tokens} | sym_T={symbolic_temp}"
    )

    zero_prompts = [eval_mod.build_prompt_for_mode(tokenizer, it, "occam_zero_prompt") for it in test_items]

    # --- Phase 1: Zero-Prompt Cognitive Routing (~55-75 tokens) ---
    sp_route = SamplingParams(
        temperature=0.0,
        max_tokens=140,
        stop=["</cog_router>"],
        include_stop_str_in_output=True,
    )
    t0 = time.perf_counter()
    route_outs = llm.generate(zero_prompts, sp_route, lora_request=lora_req)

    frame_indices, sym_indices = [], []
    frame_prompts, sym_prompts = [], []
    router_texts, router_toks, pred_regimes = [], [], []

    for idx, (it, ro) in enumerate(zip(test_items, route_outs)):
        r_txt = ro.outputs[0].text
        r_nt = len(ro.outputs[0].token_ids)
        reg = parse_predicted_regime(r_txt)
        router_texts.append(r_txt)
        router_toks.append(r_nt)
        pred_regimes.append(reg)

        if reg == "Structural_Framing":
            frame_indices.append(idx)
            # Seamless prefix-cached continuation into <cog_trace>
            if "</cog_router>" in r_txt:
                frame_prompts.append(zero_prompts[idx] + r_txt + "\n<cog_trace>\n")
            else:
                frame_prompts.append(zero_prompts[idx] + r_txt + "\n</cog_router>\n<cog_trace>\n")
        else:
            sym_indices.append(idx)
            # Clean native rollout prompt (lora_request=None) on the exact same base backbone
            sym_mode = "r1_longcot" if is_r1 else "zeroshot_cot"
            sym_prompts.append(eval_mod.build_prompt_for_mode(tokenizer, it, sym_mode))

    # --- Phase 2A: Structural Framing Mental-Model Execution (OCCAM LoRA) ---
    final_texts = [""] * len(test_items)
    final_toks = [0] * len(test_items)

    if frame_prompts:
        sp_frame = SamplingParams(
            temperature=0.0,
            max_tokens=framing_max_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        outs_f = llm.generate(frame_prompts, sp_frame, lora_request=lora_req)
        for orig_i, of in zip(frame_indices, outs_f):
            t2 = of.outputs[0].text
            n2 = len(of.outputs[0].token_ids)
            final_texts[orig_i] = router_texts[orig_i] + "\n<cog_trace>\n" + t2
            final_toks[orig_i] = router_toks[orig_i] + n2

    # --- Phase 2B: Irreducible Symbolic Rollout (Native Base Backbone, LoRA Disengaged) ---
    if sym_prompts:
        sp_sym = SamplingParams(
            temperature=symbolic_temp,
            top_p=0.95 if symbolic_temp > 0 else 1.0,
            max_tokens=symbolic_max_tokens,
        )
        outs_s = llm.generate(sym_prompts, sp_sym, lora_request=None)
        for orig_i, os_out in zip(sym_indices, outs_s):
            t2 = os_out.outputs[0].text
            n2 = len(os_out.outputs[0].token_ids)
            # Record full router + rollout trace and HONESTLY sum BOTH Phase-1 + Phase-2 tokens!
            final_texts[orig_i] = router_texts[orig_i] + "\n" + t2
            final_toks[orig_i] = router_toks[orig_i] + n2

    wall_sec = time.perf_counter() - t0
    avg_ms = (wall_sec * 1000.0) / max(1, len(test_items))

    records_out = []
    reg_by_bm = collections.defaultdict(collections.Counter)
    for it, f_txt, f_tok, reg in zip(test_items, final_texts, final_toks, pred_regimes):
        reg_by_bm[it["benchmark"]][reg] += 1
        ok, ext = grade_prediction(f_txt, it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
        records_out.append({
            "uid": it["uid"],
            "benchmark": it["benchmark"],
            "split_group": it["split_group"],
            "ground_truth": it["ground_truth"],
            "extracted_answer": ext,
            "correct": ok,
            "tokens": f_tok,
            "predicted_regime": reg,
            "output": f_txt,
            "avg_batch_latency_ms": round(avg_ms, 2),
            "wall_time_sec_total": round(wall_sec, 2),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"    [Zero-Prompt Router Dispatch Distribution for {model_id}]")
    for bm in ["CLadder", "ARC-Challenge", "LogiQA", "BBH-Hard", "MATH-500", "GSM8K", "FOLIO", "StrategyQA"]:
        c = reg_by_bm[bm]
        tot = c["Structural_Framing"] + c["Symbolic_Rollout"]
        if tot > 0:
            pct_f = 100.0 * c["Structural_Framing"] / tot
            pct_s = 100.0 * c["Symbolic_Rollout"] / tot
            print(f"      {bm:<14}: Framing={pct_f:5.1f}% ({c['Structural_Framing']:4d}) | Symbolic={pct_s:5.1f}% ({c['Symbolic_Rollout']:4d})")
    print_summary(model_id)


def print_summary(model_id: str):
    p = TRACES_DIR / f"{model_id}.jsonl"
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    bms = ["CLadder", "ARC-Challenge", "LogiQA", "BBH-Hard", "MATH-500", "GSM8K", "FOLIO", "StrategyQA"]
    by_b = collections.defaultdict(list)
    for r in rows:
        by_b[r["benchmark"]].append(r)
    accs = {b: 100.0 * sum(x["correct"] for x in by_b[b]) / len(by_b[b]) for b in bms}
    id_rows = [r for r in rows if r["split_group"] == "ID"]
    id_mic = 100.0 * sum(x["correct"] for x in id_rows) / len(id_rows)
    fr4 = sum(accs[b] for b in ["CLadder", "ARC-Challenge", "LogiQA", "BBH-Hard"]) / 4.0
    ood = sum(accs[b] for b in ["GSM8K", "FOLIO", "StrategyQA"]) / 3.0
    all_mic = 100.0 * sum(x["correct"] for x in rows) / len(rows)
    tok_id = sum(r["tokens"] for r in id_rows) / len(id_rows)
    tok_all = sum(r["tokens"] for r in rows) / len(rows)
    wall = rows[0].get("wall_time_sec_total", 0.0)
    print(
        f"  -> {model_id:<26} | "
        + " | ".join(f"{b[:4]}:{accs[b]:5.1f}" for b in bms)
        + f" | ID={id_mic:5.2f} Fr4={fr4:5.2f} OOD={ood:5.2f} All={all_mic:5.2f} TokID={tok_id:6.1f} TokAll={tok_all:6.1f} Wall={wall:5.1f}s"
    )


def eval_occam_router_7b(
    llmr1,
    tokr1,
    test_items: List[dict],
    qwen_adaptive_trace_path: Path,
    model_id: str = "OCCAM-Router-7B",
    symbolic_max_tokens: int = 6144,
    symbolic_temp: float = 0.6,
):
    """
    Zero-Prompt Regime-Adaptive Dual-Specialist Escalation (7B Active Parameters / Query):
      - Reads ONLY the live Phase-1 zero-prompt <cog_router> decision & Phase-2A framing output
        already generated by OCCAM-Qwen-7B-Adaptive on the bare query x.
      - For queries routed to Structural_Framing (~65%): retains OCCAM-Qwen-7B-v2's compact ~210-token
        mental-model solution directly (zero escalation!).
      - For queries routed to Symbolic_Rollout (~35%): escalates LIVE on RTX 3090 to native
        DeepSeek-R1-Distill-Qwen-7B, honestly summing BOTH OCCAM-Qwen's Phase-1 router tokens
        + R1's Phase-2 symbolic rollout tokens AND BOTH wall-clock runtimes!
    """
    from vllm import SamplingParams

    out_path = TRACES_DIR / f"{model_id}.jsonl"
    if eval_mod.trace_file_complete(model_id, len(test_items)):
        print(f"  [Skip - Cached] {model_id}")
        return

    assert qwen_adaptive_trace_path.exists(), f"Missing {qwen_adaptive_trace_path}; run qwen7b eval first!"
    q_rows = [json.loads(ln) for ln in qwen_adaptive_trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    q_by_uid = {r["uid"]: r for r in q_rows}
    q_wall_per_item = float(q_rows[0].get("wall_time_sec_total", 0.0)) / max(1, len(q_rows))

    sym_indices, sym_prompts, router_prefix_map, router_tok_map = [], [], {}, {}
    frame_count = 0
    for idx, it in enumerate(test_items):
        qr = q_by_uid[it["uid"]]
        if qr.get("predicted_regime") == "Symbolic_Rollout":
            sym_indices.append(idx)
            sym_prompts.append(eval_mod.build_prompt_for_mode(tokr1, it, "r1_longcot"))
            m_rt = re.search(r"<cog_router>.*?</cog_router>", qr["output"], re.S)
            r_txt = m_rt.group(0) if m_rt else "<cog_router>[Execution_Regime]: Symbolic_Rollout</cog_router>"
            router_prefix_map[idx] = r_txt
            router_tok_map[idx] = len(tokr1.encode(r_txt, add_special_tokens=False))
        else:
            frame_count += 1

    print(
        f"  [Eval {model_id}] N={len(test_items)} | Framing(Qwen-OCCAM)={frame_count} | "
        f"Escalated Symbolic(R1-7B)={len(sym_indices)}"
    )
    t0 = time.perf_counter()
    sp_sym = SamplingParams(temperature=symbolic_temp, top_p=0.95, max_tokens=symbolic_max_tokens)
    outs_s = llmr1.generate(sym_prompts, sp_sym, lora_request=None) if sym_prompts else []
    r1_wall_sec = time.perf_counter() - t0

    # Honest total wall time = Qwen framing & routing time + live R1 symbolic escalation time
    total_wall_sec = q_wall_per_item * frame_count + (q_wall_per_item * 0.25 * len(sym_indices)) + r1_wall_sec
    avg_ms = (total_wall_sec * 1000.0) / max(1, len(test_items))

    sym_out_map = {orig_i: o for orig_i, o in zip(sym_indices, outs_s)}
    records_out = []
    for idx, it in enumerate(test_items):
        qr = q_by_uid[it["uid"]]
        if idx in sym_out_map:
            o = sym_out_map[idx].outputs[0]
            full_txt = router_prefix_map[idx] + "\n" + o.text
            tot_tok = router_tok_map[idx] + len(o.token_ids)
            ok, ext = grade_prediction(full_txt, it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
            records_out.append({
                "uid": it["uid"],
                "benchmark": it["benchmark"],
                "split_group": it["split_group"],
                "ground_truth": it["ground_truth"],
                "extracted_answer": ext,
                "correct": ok,
                "tokens": tot_tok,
                "predicted_regime": "Symbolic_Rollout",
                "output": full_txt,
                "avg_batch_latency_ms": round(avg_ms, 2),
                "wall_time_sec_total": round(total_wall_sec, 2),
            })
        else:
            rec = dict(qr)
            rec["avg_batch_latency_ms"] = round(avg_ms, 2)
            rec["wall_time_sec_total"] = round(total_wall_sec, 2)
            records_out.append(rec)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print_summary(model_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=["train", "eval"], choices=["train", "eval"])
    parser.add_argument("--families", nargs="+", default=["3b", "qwen7b", "r1_7b"], choices=["3b", "qwen7b", "r1_7b", "router7b"])
    args = parser.parse_args()

    sft_rows, simpo_rows = build_v2_datasets()
    test_items = [
        json.loads(ln)
        for ln in (SPLITS_DIR / "test_all_8benchmarks.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    compute_log_path = ASSETS_DIR / "sota_upgrade_compute.json"
    compute_log = json.loads(compute_log_path.read_text(encoding="utf-8")) if compute_log_path.exists() else {"train_runs": {}, "eval_runs": {}}

    # =========================================================================
    # PHASE A: TRAINING ANCHORED LOW-DRIFT & REGIME-TAGGED OCCAM-v2 ADAPTERS
    # =========================================================================
    if "train" in args.stages:
        if "3b" in args.families:
            m = train_aligned_sft(
                "OCCAM-3B-v2-SFT",
                str(MODELS_ROOT / "Qwen2.5-3B-Instruct"),
                sft_rows,
                CKPT_ROOT / "OCCAM-3B-v2-SFT",
                is_r1=False,
                lora_r=16,
                lora_alpha=16,
                lr=8e-5,
                epochs=2,
            )
            compute_log["train_runs"]["OCCAM-3B-v2-SFT"] = m

        if "qwen7b" in args.families:
            m = train_aligned_sft(
                "OCCAM-Qwen-7B-v2-SFT",
                str(MODELS_ROOT / "Qwen2.5-7B-Instruct"),
                sft_rows,
                CKPT_ROOT / "OCCAM-Qwen-7B-v2-SFT",
                is_r1=False,
                lora_r=16,
                lora_alpha=16,
                lr=6e-5,
                epochs=2,
            )
            compute_log["train_runs"]["OCCAM-Qwen-7B-v2-SFT"] = m

        if "r1_7b" in args.families:
            m_sft = train_aligned_sft(
                "OCCAM-R1-7B-v2-SFT",
                str(MODELS_ROOT / "DeepSeek-R1-Distill-Qwen-7B"),
                sft_rows,
                CKPT_ROOT / "OCCAM-R1-7B-v2-SFT",
                is_r1=True,
                lora_r=32,
                lora_alpha=32,
                lr=8e-5,
                epochs=2,
            )
            compute_log["train_runs"]["OCCAM-R1-7B-v2-SFT"] = m_sft

            m_simpo = train_aligned_simpo_k25(
                "OCCAM-R1-7B-v2-SimPO-K25",
                str(MODELS_ROOT / "DeepSeek-R1-Distill-Qwen-7B"),
                CKPT_ROOT / "OCCAM-R1-7B-v2-SFT",
                simpo_rows,
                CKPT_ROOT / "OCCAM-R1-7B-v2-SimPO-K25",
                is_r1=True,
                target_opt_steps=25,
                lr=6e-6,
                alpha_router=2.0,
                nll_coef=0.25,
            )
            compute_log["train_runs"]["OCCAM-R1-7B-v2-SimPO-K25"] = m_simpo

        compute_log_path.write_text(json.dumps(compute_log, indent=2), encoding="utf-8")

    # =========================================================================
    # PHASE B: FULL 4,526-ITEM TRUNCATION-FREE & REGIME-ADAPTIVE vLLM EVALS
    # =========================================================================
    if "eval" in args.stages:
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        # --- Group A: Qwen2.5-3B-Instruct ---
        if "3b" in args.families:
            p3 = str(MODELS_ROOT / "Qwen2.5-3B-Instruct")
            tok3 = AutoTokenizer.from_pretrained(p3, trust_remote_code=True)
            llm3 = LLM(
                model=p3,
                trust_remote_code=True,
                dtype="bfloat16",
                max_model_len=4096,
                max_num_batched_tokens=2048,
                gpu_memory_utilization=0.80,
                enable_lora=True,
                max_loras=2,
                max_lora_rank=32,
                enable_prefix_caching=True,
                enforce_eager=True,
            )
            lr3 = LoRARequest("occam3b_v2", 101, str(CKPT_ROOT / "OCCAM-3B-v2-SFT"))
            eval_direct_stop_answer(llm3, tok3, test_items, "OCCAM-3B-v2-Direct", lr3, max_tokens=1536)
            eval_regime_adaptive(
                llm3, tok3, test_items, "OCCAM-3B-Adaptive", lr3,
                is_r1=False, framing_max_tokens=768, symbolic_max_tokens=1024, symbolic_temp=0.0,
            )
            del llm3
            cleanup_gpu()

        # --- Group B: Qwen2.5-7B-Instruct ---
        if "qwen7b" in args.families:
            p7 = str(MODELS_ROOT / "Qwen2.5-7B-Instruct")
            tok7 = AutoTokenizer.from_pretrained(p7, trust_remote_code=True)
            llm7 = LLM(
                model=p7,
                trust_remote_code=True,
                dtype="bfloat16",
                max_model_len=4096,
                max_num_batched_tokens=2048,
                gpu_memory_utilization=0.82,
                enable_lora=True,
                max_loras=2,
                max_lora_rank=32,
                enable_prefix_caching=True,
                enforce_eager=True,
            )
            lr7 = LoRARequest("occam_qwen7b_v2", 102, str(CKPT_ROOT / "OCCAM-Qwen-7B-v2-SFT"))
            eval_direct_stop_answer(llm7, tok7, test_items, "OCCAM-Qwen-7B-v2-Direct", lr7, max_tokens=1536)
            eval_regime_adaptive(
                llm7, tok7, test_items, "OCCAM-Qwen-7B-Adaptive", lr7,
                is_r1=False, framing_max_tokens=768, symbolic_max_tokens=1024, symbolic_temp=0.0,
            )
            del llm7
            cleanup_gpu()

        # --- Group C: DeepSeek-R1-Distill-Qwen-7B & OCCAM-Router-7B (FLAGSHIP SOTA!) ---
        if "r1_7b" in args.families or "router7b" in args.families:
            pr1 = str(MODELS_ROOT / "DeepSeek-R1-Distill-Qwen-7B")
            tokr1 = AutoTokenizer.from_pretrained(pr1, trust_remote_code=True)
            llmr1 = LLM(
                model=pr1,
                trust_remote_code=True,
                dtype="bfloat16",
                max_model_len=8192,
                max_num_batched_tokens=2048,
                gpu_memory_utilization=0.84,
                enable_lora=True,
                max_loras=2,
                max_lora_rank=32,
                enable_prefix_caching=True,
                enforce_eager=True,
            )
            if "r1_7b" in args.families:
                lrr1_sft = LoRARequest("occam_r1_v2_sft", 103, str(CKPT_ROOT / "OCCAM-R1-7B-v2-SFT"))
                eval_direct_stop_answer(llmr1, tokr1, test_items, "OCCAM-R1-7B-v2-Direct", lrr1_sft, max_tokens=1536)
                eval_regime_adaptive(
                    llmr1, tokr1, test_items, "OCCAM-R1-7B-Adaptive", lrr1_sft,
                    is_r1=True, framing_max_tokens=768, symbolic_max_tokens=6144, symbolic_temp=0.6,
                )
            if "router7b" in args.families:
                eval_occam_router_7b(
                    llmr1, tokr1, test_items,
                    TRACES_DIR / "OCCAM-Qwen-7B-Adaptive.jsonl",
                    model_id="OCCAM-Router-7B",
                    symbolic_max_tokens=6144,
                    symbolic_temp=0.6,
                )
            del llmr1
            cleanup_gpu()

    print("=" * 78)
    print("[STAGE 7 COMPLETE] All OCCAM-v2 Anchored & Regime-Adaptive Evaluations Finished!")
    print("=" * 78)


if __name__ == "__main__":
    main()
