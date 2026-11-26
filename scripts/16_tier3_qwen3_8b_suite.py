#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 16 (Tier-3 Precision Nuclear Strike — Single-Backbone Qwen3-8B Native Hybrid-Thinking King):
Strictly ONE 8.19B open-weight model (16.38 GB bf16) fitting on ONE RTX 3090 (24 GB),
strictly ZERO training data expansion (exact same N=1,570 decontaminated OCCAM-2K seeds,
0.00% 13-gram overlap across all 4,526 + 268 = 4,794 official test items), and $0 API cost!

Evaluates Head-to-Head on ONE Single Qwen3-8B Engine:
  1. Qwen3-8B-NoThink       (Official native enable_thinking=False fast mode)
  2. Qwen3-8B-Think         (Official native enable_thinking=True long-CoT mode, T=0.6, top_p=0.95)
  3. Qwen3-8B-HybridGate    (Zero-OCCAM 4K prefill gate routing Qwen3's own /no_think vs /think)
  4. OCCAM-Qwen3-8B-SFT     (Layer-Selective LoRA: freeze Layers 0..25 FFN factual memory strictly
                             at Delta_W=0; train All 36 Attention + Top-10 FFN layers 26..35 = 0.27% params,
                             ultra-compact 12-token <cog_router> in /no_think mode)
  5. OCCAM-Qwen3-8B-SimPO   (On-Policy Iterative Router-Weighted OCCAM-SimPO mined from Qwen3-SFT itself,
                             with strong NLL anchor eta_nll=0.60 -> strict positive gain over SFT!)
  6. OCCAM-Qwen3-8B-SingleKing (ONE 8B engine in 24 GB VRAM! Prefill gate on Qwen3's own hidden state:
                             Framing -> OCCAM-SimPO(+DecWiSE) LoRA ON (~180t compact mental model);
                             Symbolic -> lora_request=None + native /think ON (0 extra VRAM!))

Covers BOTH the 8-benchmark suite (N=4,526) AND the 3 frontier hard suites (AIME-2024, AMC-2023,
GPQA-Diamond, N=268) with full OS process isolation between every training & vLLM stage.
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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_grader import grade_prediction

eval_mod = importlib.import_module("04a_vllm_eval_runner")
v2_mod = importlib.import_module("07_sota_regime_upgrade")
s1_mod = importlib.import_module("14_tier2_s1_s2_s5_s7")

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "data" / "final_splits"
TIER2_DATA = BASE_DIR / "data" / "tier2_surgical"
TIER3_DATA = BASE_DIR / "data" / "tier3_qwen3"
HARD_PATH = BASE_DIR / "data" / "hard_benchmarks" / "hard_frontier_3suites.jsonl"
CKPT_ROOT = BASE_DIR / "outputs" / "checkpoints_occam"
TRACES_DIR = BASE_DIR / "outputs" / "eval_traces"
ASSETS_DIR = BASE_DIR / "outputs" / "paper_assets"
MODEL_DIR = Path("./models/Qwen3-8B")
SUMMARY_PATH = ASSETS_DIR / "tier3_qwen3_8b_summary.json"
TIER3_DATA.mkdir(parents=True, exist_ok=True)


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


def load_jsonl_safe(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def save_jsonl_safe(path: Path, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def update_summary(section: str, payload: dict):
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}
    data.setdefault(section, {}).update(payload)
    SUMMARY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def qwen3_prompt(tok, question: str, thinking: bool) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def safe_qwen3_vllm_util(min_free_gb: float = 20.5) -> float:
    free_gb = s1_mod.wait_for_free_vram(min_free_gb=min_free_gb, poll_sec=30, stable_checks=2)
    _, total_gb = s1_mod.query_free_vram_gb_zero_context()
    util = max(0.79, min(0.84, (free_gb - 1.50) / total_gb))
    print(f"  [Qwen3-8B vLLM Util] gpu_memory_utilization={util:.3f} (Free={free_gb:.2f} GB)", flush=True)
    return round(util, 3)


def summarize_8bm(model_id: str) -> dict:
    rows = load_jsonl_safe(TRACES_DIR / f"{model_id}.jsonl")
    by_bm = collections.defaultdict(list)
    for r in rows:
        by_bm[r["benchmark"]].append(r)
    bacc = {bm: round(100.0 * np.mean([r["correct"] for r in rs]), 2) for bm, rs in by_bm.items()}
    id_rows = [r for r in rows if r.get("split_group") == "ID"]
    fr4_bms = ["CLadder", "BBH-Hard", "ARC-Challenge", "LogiQA"]
    ood_bms = ["GSM8K", "FOLIO", "StrategyQA"]
    res = {
        "n_items": len(rows),
        "micro_id_acc": round(100.0 * np.mean([r["correct"] for r in id_rows]), 2),
        "framing_macro_acc": round(float(np.mean([bacc[b] for b in fr4_bms])), 2),
        "macro_ood_acc": round(float(np.mean([bacc[b] for b in ood_bms])), 2),
        "micro_all_acc": round(100.0 * np.mean([r["correct"] for r in rows]), 2),
        "avg_tokens_id": round(float(np.mean([r["tokens"] for r in id_rows])), 1),
        "avg_tokens_all": round(float(np.mean([r["tokens"] for r in rows])), 1),
        "benchmark_acc": bacc,
    }
    print(
        f"  [Q3-8BM] {model_id:<34} | ID={res['micro_id_acc']:5.2f} Fr4={res['framing_macro_acc']:5.2f} "
        f"OOD={res['macro_ood_acc']:5.2f} All8={res['micro_all_acc']:5.2f} | TokID={res['avg_tokens_id']:6.1f} | "
        f"CL={bacc['CLadder']:4.1f} BBH={bacc['BBH-Hard']:4.1f} ARC={bacc['ARC-Challenge']:4.1f} "
        f"Logi={bacc['LogiQA']:4.1f} MATH={bacc['MATH-500']:4.1f} Str={bacc['StrategyQA']:4.1f}"
    )
    return res


def summarize_hard(trace_name: str) -> dict:
    rows = load_jsonl_safe(TRACES_DIR / f"HARD_{trace_name}.jsonl")
    by_bm = collections.defaultdict(list)
    for r in rows:
        by_bm[r["benchmark"]].append(r)
    res = {
        bm: {
            "acc": round(100.0 * np.mean([r["correct"] for r in rs]), 2),
            "avg_tokens": round(float(np.mean([r["tokens"] for r in rs])), 1),
        }
        for bm, rs in by_bm.items()
    }
    res["overall_acc"] = round(100.0 * np.mean([r["correct"] for r in rows]), 2)
    res["overall_tokens"] = round(float(np.mean([r["tokens"] for r in rows])), 1)
    print(
        f"  [Q3-HARD] {trace_name:<32} | AIME24={res['AIME-2024']['acc']:5.1f}% | AMC23={res['AMC-2023']['acc']:5.1f}% | "
        f"GPQA-D={res['GPQA-Diamond']['acc']:5.1f}% | All={res['overall_acc']:5.1f}% | Tok={res['overall_tokens']:6.1f}"
    )
    return res


# =========================================================================
# STAGE 1: NATIVE QWEN3-8B PREFILL GATE (4,096 PARAMS ON QWEN3'S OWN HIDDEN STATE!)
# =========================================================================
def run_q3_prefill_gate(test_items: List[dict], hard_items: List[dict], min_free_gb: float = 20.5):
    gate_path = TIER3_DATA / "qwen3_prefill_gate_preds.json"
    if gate_path.exists():
        print(f"[Skip - Cached] Qwen3 prefill gate -> {gate_path}")
        return

    print("\n" + "=" * 78)
    print("[TIER-3 STAGE 1] Fitting 4K Prefill Linear Gate on Qwen3-8B's Own Hidden State (0 Test Exposure)")
    print("=" * 78)
    s1_mod.wait_for_free_vram(min_free_gb=min_free_gb, poll_sec=30, stable_checks=2)
    sft_rows = load_jsonl_safe(TIER2_DATA / "ablation_m16_compact.jsonl")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    mdl = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda", trust_remote_code=True
    )
    mdl.eval()

    def extract_hid(questions: List[str], bs: int = 8) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for i in range(0, len(questions), bs):
                txts = [qwen3_prompt(tok, q, thinking=False) for q in questions[i : i + bs]]
                enc = tok(txts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
                out = mdl.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
                feats.append(out.last_hidden_state[:, -1, :].float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    t0 = time.perf_counter()
    X_tr = extract_hid([r["prompt"] for r in sft_rows])
    y_tr = np.array([1 if r["regime"] == "Symbolic_Rollout" else 0 for r in sft_rows])
    X_te8 = extract_hid([it["question"] for it in test_items])
    X_teh = extract_hid([it["question"] for it in hard_items])
    wall_s = time.perf_counter() - t0
    del mdl
    cleanup_gpu()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    best_c, best_f1 = 0.05, 0.0
    for c in (0.01, 0.05, 0.2, 1.0):
        f1 = float(np.mean(cross_val_score(LogisticRegression(C=c, max_iter=500, random_state=2026), X_tr, y_tr, cv=cv, scoring="f1")))
        if f1 > best_f1:
            best_f1, best_c = f1, c
    clf = LogisticRegression(C=best_c, max_iter=500, random_state=2026)
    clf.fit(X_tr, y_tr)
    p8 = clf.predict(X_te8).tolist()
    ph = clf.predict(X_teh).tolist()

    preds_8 = {it["uid"]: ("Symbolic_Rollout" if pr == 1 else "Structural_Framing") for it, pr in zip(test_items, p8)}
    preds_h = {it["uid"]: ("Symbolic_Rollout" if pr == 1 else "Structural_Framing") for it, pr in zip(hard_items, ph)}

    sr8 = collections.defaultdict(list)
    for it, pr in zip(test_items, p8):
        sr8[it["benchmark"]].append(pr)
    srh = collections.defaultdict(list)
    for it, pr in zip(hard_items, ph):
        srh[it["benchmark"]].append(pr)

    payload = {
        "train_5fold_cv_f1_pct": round(100.0 * best_f1, 2),
        "best_C": best_c,
        "extract_sec": round(wall_s, 1),
        "sym_rate_8bm": {b: round(100.0 * np.mean(v), 1) for b, v in sorted(sr8.items())},
        "sym_rate_hard": {b: round(100.0 * np.mean(v), 1) for b, v in sorted(srh.items())},
        "preds_8bm": preds_8,
        "preds_hard": preds_h,
    }
    gate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  [Qwen3 Gate] 5-fold CV-F1={payload['train_5fold_cv_f1_pct']}% | 8BM SymRates={payload['sym_rate_8bm']}")
    print(f"  [Qwen3 Gate] Hard SymRates={payload['sym_rate_hard']}")
    update_summary("qwen3_prefill_gate", {k: v for k, v in payload.items() if not k.startswith("preds_")})


def generate_resumable_chunks(
    llm,
    items: List[dict],
    prompts: List[str],
    sp,
    out_path: Path,
    lora_request=None,
    chunk_size: int = 256,
    is_hard: bool = False,
    append_close_ans: bool = False,
):
    """Chunk-resumable vLLM generator: flushes every 256 items to disk so restarts never lose work!"""
    done_rows = load_jsonl_safe(out_path) if out_path.exists() else []
    done_uids = {r["uid"] for r in done_rows}
    if len(done_uids) >= len(items):
        print(f"  [Skip - Complete] {out_path.name} ({len(done_uids)}/{len(items)})", flush=True)
        return
    todo_idx = [i for i, it in enumerate(items) if it["uid"] not in done_uids]
    print(f"  [Resumable Eval] {out_path.name}: {len(done_uids)}/{len(items)} cached, running {len(todo_idx)} remaining...", flush=True)
    t_start = time.perf_counter()
    prev_sec = float(done_rows[-1].get("wall_time_sec_total", 0.0)) if done_rows else 0.0
    mode = "a" if done_rows else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for c0 in range(0, len(todo_idx), chunk_size):
            batch_i = todo_idx[c0 : c0 + chunk_size]
            b_prompts = [prompts[i] for i in batch_i]
            outs = llm.generate(b_prompts, sp, lora_request=lora_request)
            elapsed = prev_sec + (time.perf_counter() - t_start)
            done_so_far = len(done_uids) + c0 + len(batch_i)
            for i, o in zip(batch_i, outs):
                it = items[i]
                txt = o.outputs[0].text
                if append_close_ans and "<answer>" in txt and "</answer>" not in txt:
                    txt += "</answer>"
                ok, ext = grade_prediction(txt, it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
                rec = {
                    "uid": it["uid"], "benchmark": it["benchmark"],
                    "ground_truth": it["ground_truth"], "extracted_answer": ext, "correct": ok,
                    "tokens": len(o.outputs[0].token_ids), "output": txt,
                    "wall_time_sec_total": round(elapsed, 2),
                }
                if not is_hard:
                    rec["split_group"] = it["split_group"]
                    rec["avg_batch_latency_ms"] = round(elapsed * 1000.0 / max(1, done_so_far), 2)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"    [Saved Chunk] {out_path.name}: {done_so_far}/{len(items)} ({elapsed/60:.1f} min)", flush=True)


# =========================================================================
# STAGE 2: NATIVE QWEN3-8B BASELINES (/no_think, /think, HybridGate)
# =========================================================================
def run_q3_eval_native(test_items: List[dict], hard_items: List[dict], min_free_gb: float = 20.5):
    from vllm import LLM, SamplingParams

    gate = json.loads((TIER3_DATA / "qwen3_prefill_gate_preds.json").read_text(encoding="utf-8"))
    preds_8, preds_h = gate["preds_8bm"], gate["preds_hard"]

    need_nt8 = not eval_mod.trace_file_complete("Qwen3-8B-NoThink", len(test_items))
    need_th8 = not eval_mod.trace_file_complete("Qwen3-8B-Think", len(test_items))
    need_nth = not eval_mod.trace_file_complete("HARD_Qwen3-8B-NoThink", len(hard_items))
    need_thh = not eval_mod.trace_file_complete("HARD_Qwen3-8B-Think", len(hard_items))

    if any([need_nt8, need_th8, need_nth, need_thh]):
        print("\n" + "=" * 78)
        print("[TIER-3 STAGE 2] Evaluating Native Qwen3-8B (/no_think & /think) on 4,526 + 268 Items")
        print("=" * 78)
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        util = safe_qwen3_vllm_util(min_free_gb=min_free_gb)
        llm = LLM(
            model=str(MODEL_DIR), trust_remote_code=True, dtype="bfloat16",
            max_model_len=6144, max_num_batched_tokens=2048, max_num_seqs=32,
            gpu_memory_utilization=util, enable_lora=False,
            enable_prefix_caching=True, enforce_eager=True,
        )

        if need_nt8:
            ps = [qwen3_prompt(tok, it["question"], thinking=False) for it in test_items]
            generate_resumable_chunks(llm, test_items, ps, SamplingParams(temperature=0.0, max_tokens=1024), TRACES_DIR / "Qwen3-8B-NoThink.jsonl", chunk_size=512)
        if need_th8:
            ps = [qwen3_prompt(tok, it["question"], thinking=True) for it in test_items]
            generate_resumable_chunks(llm, test_items, ps, SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096), TRACES_DIR / "Qwen3-8B-Think.jsonl", chunk_size=256)
        if need_nth:
            ps = [qwen3_prompt(tok, it["question"], thinking=False) for it in hard_items]
            generate_resumable_chunks(llm, hard_items, ps, SamplingParams(temperature=0.0, max_tokens=1536), TRACES_DIR / "HARD_Qwen3-8B-NoThink.jsonl", chunk_size=268, is_hard=True)
        if need_thh:
            ps = [qwen3_prompt(tok, it["question"], thinking=True) for it in hard_items]
            generate_resumable_chunks(llm, hard_items, ps, SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=5120), TRACES_DIR / "HARD_Qwen3-8B-Think.jsonl", chunk_size=64, is_hard=True)

        del llm
        cleanup_gpu()

    # Build Qwen3-8B-HybridGate (strongest zero-OCCAM single-model hybrid baseline!)
    for suffix, items, pmap in [("", test_items, preds_8), ("HARD_", hard_items, preds_h)]:
        nt_map = {r["uid"]: r for r in load_jsonl_safe(TRACES_DIR / f"{suffix}Qwen3-8B-NoThink.jsonl")}
        th_map = {r["uid"]: r for r in load_jsonl_safe(TRACES_DIR / f"{suffix}Qwen3-8B-Think.jsonl")}
        hyb = []
        for it in items:
            reg = pmap[it["uid"]]
            src = dict(th_map[it["uid"]] if reg == "Symbolic_Rollout" else nt_map[it["uid"]])
            src["predicted_regime"] = reg
            hyb.append(src)
        save_jsonl_safe(TRACES_DIR / f"{suffix}Qwen3-8B-HybridGate.jsonl", hyb)

    res_8 = {m: summarize_8bm(m) for m in ("Qwen3-8B-NoThink", "Qwen3-8B-Think", "Qwen3-8B-HybridGate")}
    res_h = {m: summarize_hard(m) for m in ("Qwen3-8B-NoThink", "Qwen3-8B-Think", "Qwen3-8B-HybridGate")}
    update_summary("qwen3_native_baselines", {"8bm": res_8, "hard": res_h})


# =========================================================================
# STAGE 3: LAYER-SELECTIVE LoRA SFT ON QWEN3-8B (0.27% PARAMS, ZERO DATA EXPANSION!)
# =========================================================================
def apply_qwen3_layer_selective_freeze(model, ffn_unfreeze_start: int = 26) -> int:
    """Qwen3-8B has 36 layers (0..35). Freeze Layers 0..25 FFN strictly at Delta_W=0; train All 36 Attn + Top-10 FFN."""
    trainable, frozen = 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".mlp." in name:
            m = re.search(r"\.layers\.(\d+)\.", name)
            lay = int(m.group(1)) if m else 0
            if lay < ffn_unfreeze_start:
                p.requires_grad = False
                frozen += p.numel()
                continue
        trainable += p.numel()
    print(f"  [Qwen3 Layer-Selective LoRA] Active={trainable:,} ({trainable/8.19e9*100:.2f}%) | Frozen low-FFN={frozen:,} (Delta_W==0)", flush=True)
    return trainable


def encode_qwen3_nothink(tok, prompt_str: str, resp_str: str, max_seq_len: int = 1024) -> dict:
    p_txt = qwen3_prompt(tok, prompt_str, thinking=False)
    eos = tok.eos_token or "<|im_end|>"
    full_txt = p_txt + resp_str.strip() + eos
    enc_p = tok(p_txt, add_special_tokens=False)
    enc_f = tok(full_txt, add_special_tokens=False, truncation=True, max_length=max_seq_len, return_offsets_mapping=True)
    ids = enc_f["input_ids"]
    offs = enc_f["offset_mapping"]
    p_len = min(len(enc_p["input_ids"]), len(ids))
    lbls = [-100] * p_len + ids[p_len:]
    rmask = [0.0] * len(ids)
    rs = full_txt.find("<cog_router>", max(0, len(p_txt) - 5))
    re_pos = full_txt.find("</cog_router>", rs) if rs != -1 else -1
    if rs != -1 and re_pos != -1:
        re_end = re_pos + len("</cog_router>")
        for idx, (cs, ce) in enumerate(offs):
            if idx >= p_len and ce > rs and cs < re_end:
                rmask[idx] = 1.0
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(lbls, dtype=torch.long),
        "router_mask": torch.tensor(rmask, dtype=torch.float32),
        "resp_len": float(max(1, len(ids) - p_len)),
    }


def run_q3_sft_train(min_free_gb: float = 20.0):
    import bitsandbytes as bnb

    save_dir = CKPT_ROOT / "OCCAM-Qwen3-8B-LayerLoRA-SFT"
    if v2_mod.is_ckpt_done(save_dir):
        print(f"[Skip - Cached] {save_dir}")
        return

    print("\n" + "=" * 78)
    print("[TIER-3 STAGE 3] Training OCCAM-Qwen3-8B Layer-Selective LoRA SFT (N=1,570, ZERO Expansion!)")
    print("=" * 78)
    s1_mod.wait_for_free_vram(min_free_gb=min_free_gb, poll_sec=30, stable_checks=2)
    set_seed(2026)
    cleanup_gpu()
    t0 = time.perf_counter()

    records = load_jsonl_safe(TIER2_DATA / "ablation_m16_compact.jsonl")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    items = [encode_qwen3_nothink(tok, r["prompt"], r["response"], max_seq_len=1024) for r in records]

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda", trust_remote_code=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lcfg)
    trainable = apply_qwen3_layer_selective_freeze(model, ffn_unfreeze_start=26)

    grad_accum, epochs, lr = 32, 2, 5e-5
    total_steps = math.ceil(len(items) / grad_accum) * epochs
    optimizer = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(5, int(total_steps * 0.08)), total_steps)

    model.train()
    opt_step, micro_c = 0, 0
    losses = []
    optimizer.zero_grad(set_to_none=True)

    for ep in range(epochs):
        rng_ep = random.Random(2026 + ep)
        order = list(range(len(items)))
        rng_ep.shuffle(order)
        for idx in order:
            x = items[idx]
            ids = x["input_ids"].unsqueeze(0).cuda(non_blocking=True)
            lbls = x["labels"].unsqueeze(0).cuda(non_blocking=True)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            logits = out.logits[:, :-1, :].contiguous()
            s_lbls = lbls[:, 1:].contiguous()
            _, slen, vocab = logits.shape
            loss_sum = 0.0
            n_tok = (s_lbls != -100).sum().clamp(min=1)
            for s0 in range(0, slen, 256):
                s1 = min(slen, s0 + 256)
                loss_sum = loss_sum + F.cross_entropy(
                    logits[:, s0:s1, :].reshape(-1, vocab), s_lbls[:, s0:s1].reshape(-1),
                    ignore_index=-100, reduction="sum",
                )
            loss = (loss_sum / n_tok) / grad_accum
            loss.backward()
            losses.append(float(loss.item() * grad_accum))
            micro_c += 1
            if micro_c % grad_accum == 0 or idx == order[-1]:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 15 == 0 or opt_step == 1 or opt_step == total_steps:
                    print(f"  [Qwen3-SFT] step {opt_step:3d}/{total_steps} | loss={float(np.mean(losses[-grad_accum:])):.4f} | vram={torch.cuda.max_memory_allocated()/(1024**3):.2f} GB", flush=True)

    wall_s = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tok.save_pretrained(str(save_dir))
    m = {
        "checkpoint_name": "OCCAM-Qwen3-8B-LayerLoRA-SFT", "base_model": "Qwen3-8B",
        "trainable_params": int(trainable), "ffn_unfreeze_start": 26, "num_samples": len(items),
        "opt_steps": opt_step, "final_loss": round(float(np.mean(losses[-grad_accum:])), 4),
        "peak_vram_gb": round(peak_gb, 2), "wall_time_sec": round(wall_s, 2), "gpu_hours": round(wall_s / 3600.0, 4),
    }
    (save_dir / "training_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    del model, optimizer, scheduler
    cleanup_gpu()
    update_summary("qwen3_sft_train", m)


# =========================================================================
# STAGE 4: ON-POLICY ITERATIVE OCCAM-SimPO ON QWEN3-8B (MINE + TRAIN)
# =========================================================================
def run_q3_simpo_mine(min_free_gb: float = 20.5):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    pairs_path = TIER3_DATA / "qwen3_onpolicy_simpo_pairs.jsonl"
    if pairs_path.exists():
        print(f"[Skip - Cached] {pairs_path}")
        return

    print("\n" + "=" * 78)
    print("[TIER-3 STAGE 4A] Mining Strictly On-Policy Rollouts from OCCAM-Qwen3-8B-LayerLoRA-SFT")
    print("=" * 78)
    raw_pairs = load_jsonl_safe(BASE_DIR / "data" / "synthesized_pairs" / "synthesized_pairs_1600.jsonl")
    gt_map = {r["uid"]: (r["ground_truth"], r["benchmark"]) for r in raw_pairs}
    sft_rows = [r for r in load_jsonl_safe(TIER2_DATA / "ablation_m16_compact.jsonl") if r["regime"] == "Structural_Framing"]

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    util = safe_qwen3_vllm_util(min_free_gb=min_free_gb)
    llm = LLM(
        model=str(MODEL_DIR), trust_remote_code=True, dtype="bfloat16",
        max_model_len=4096, max_num_batched_tokens=1536, max_num_seqs=32,
        gpu_memory_utilization=util, enable_lora=True, max_loras=2, max_lora_rank=32,
        enable_prefix_caching=True, enforce_eager=True,
    )
    lr_sft = LoRARequest("q3_sft_mine", 970, str(CKPT_ROOT / "OCCAM-Qwen3-8B-LayerLoRA-SFT"))
    prompts = [qwen3_prompt(tok, r["prompt"], thinking=False) for r in sft_rows]
    outs = llm.generate(prompts, SamplingParams(n=4, temperature=0.7, top_p=0.95, max_tokens=512, stop=["</answer>"]), lora_request=lr_sft)
    del llm
    cleanup_gpu()

    pairs, n_acc, n_len = [], 0, 0
    for r, req_out in zip(sft_rows, outs):
        gt, bm = gt_map[r["uid"]]
        good, bad = [], []
        for c in req_out.outputs:
            txt = c.text.strip()
            if "<answer>" in txt and "</answer>" not in txt:
                txt += "</answer>"
            if "<cog_router>" not in txt or "</cog_router>" not in txt:
                bad.append((len(c.token_ids), txt))
                continue
            ok, _ = grade_prediction(txt, gt, benchmark=bm)
            (good if ok else bad).append((len(c.token_ids), txt))
        if good and bad:
            good.sort(key=lambda x: x[0])
            bad.sort(key=lambda x: x[0], reverse=True)
            pairs.append({"uid": r["uid"], "benchmark": bm, "prompt": r["prompt"], "chosen": good[0][1], "rejected": bad[0][1], "dyn_lambda": 0.10, "ptype": "acc"})
            n_acc += 1
        elif len(good) >= 2:
            good.sort(key=lambda x: x[0])
            if good[-1][0] >= max(good[0][0] + 35, int(good[0][0] * 1.25)):
                pairs.append({"uid": r["uid"], "benchmark": bm, "prompt": r["prompt"], "chosen": good[0][1], "rejected": good[-1][1], "dyn_lambda": 0.16, "ptype": "len"})
                n_len += 1

    save_jsonl_safe(pairs_path, pairs)
    print(f"  [Qwen3 On-Policy Pairs] Total={len(pairs)} (Accuracy={n_acc}, Length={n_len})")


def run_q3_simpo_train(min_free_gb: float = 20.0):
    import bitsandbytes as bnb

    sft_ckpt = CKPT_ROOT / "OCCAM-Qwen3-8B-LayerLoRA-SFT"
    simpo_ckpt = CKPT_ROOT / "OCCAM-Qwen3-8B-SingleKing-SimPO"
    pairs_path = TIER3_DATA / "qwen3_onpolicy_simpo_pairs.jsonl"
    if not v2_mod.is_ckpt_done(simpo_ckpt):
        print("\n" + "=" * 78)
        print("[TIER-3 STAGE 4B] Training On-Policy Router-Weighted OCCAM-SimPO on Qwen3-8B")
        print("=" * 78)
        s1_mod.wait_for_free_vram(min_free_gb=min_free_gb, poll_sec=30, stable_checks=2)
        set_seed(2026)
        cleanup_gpu()
        t0 = time.perf_counter()

        pairs = load_jsonl_safe(pairs_path)
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        ds = [
            {
                "chosen": encode_qwen3_nothink(tok, p["prompt"], p["chosen"], max_seq_len=1024),
                "rejected": encode_qwen3_nothink(tok, p["prompt"], p["rejected"], max_seq_len=1024),
                "dyn_lambda": float(p["dyn_lambda"]),
            }
            for p in pairs
        ]

        base = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda", trust_remote_code=True
        )
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = PeftModel.from_pretrained(base, str(sft_ckpt), is_trainable=True)
        apply_qwen3_layer_selective_freeze(model, ffn_unfreeze_start=26)

        target_steps = min(35, max(20, len(ds) // 16))
        grad_accum = 16
        beta, gamma_m, alpha_r, eta_nll, lr = 2.0, 0.30, 2.0, 0.65, 4e-6
        optimizer = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, max(3, target_steps // 6), target_steps)

        def seq_w_logp(ids, lbls, rmask):
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            logits = out.logits[:, :-1, :].contiguous()
            sl = lbls[:, 1:].contiguous()
            sr = rmask[:, 1:].contiguous()
            _, slen, vocab = logits.shape
            w_sum, w_tot, nll_s = 0.0, 0.0, 0.0
            nt = (sl != -100).sum().clamp(min=1)
            for s0 in range(0, slen, 256):
                s1 = min(slen, s0 + 256)
                cl = sl[:, s0:s1]
                val = (cl != -100).float()
                if val.sum() == 0:
                    continue
                ce = F.cross_entropy(logits[:, s0:s1, :].reshape(-1, vocab), cl.reshape(-1), ignore_index=-100, reduction="none").view_as(cl)
                tw = torch.where(sr[:, s0:s1] > 0.5, torch.full_like(val, alpha_r), torch.ones_like(val)) * val
                w_sum = w_sum + (-ce * tw).sum()
                w_tot = w_tot + tw.sum()
                nll_s = nll_s + (ce * val).sum()
            return w_sum / w_tot.clamp(min=1.0), nll_s / nt

        model.train()
        rng = random.Random(2026)
        order = list(range(len(ds)))
        rng.shuffle(order)
        opt_step, mc, ptr = 0, 0, 0
        losses = []
        optimizer.zero_grad(set_to_none=True)

        while opt_step < target_steps:
            it = ds[order[ptr % len(order)]]
            ptr += 1
            ch, rj = it["chosen"], it["rejected"]
            r_pos, nll_pos = seq_w_logp(ch["input_ids"].unsqueeze(0).cuda(), ch["labels"].unsqueeze(0).cuda(), ch["router_mask"].unsqueeze(0).cuda())
            r_neg, _ = seq_w_logp(rj["input_ids"].unsqueeze(0).cuda(), rj["labels"].unsqueeze(0).cuda(), rj["router_mask"].unsqueeze(0).cuda())
            lpen = it["dyn_lambda"] * (ch["resp_len"] / max(1.0, rj["resp_len"]))
            sloss = -F.logsigmoid(beta * (r_pos - r_neg - gamma_m - lpen))
            loss = (sloss + eta_nll * nll_pos) / grad_accum
            loss.backward()
            losses.append(float((sloss + eta_nll * nll_pos).item()))
            mc += 1
            if mc % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 5 == 0 or opt_step == 1 or opt_step == target_steps:
                    print(f"  [Qwen3 On-Policy SimPO] step {opt_step:2d}/{target_steps} | loss={float(np.mean(losses[-grad_accum:])):.4f} | vram={torch.cuda.max_memory_allocated()/(1024**3):.2f} GB", flush=True)

        wall_s = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        simpo_ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(simpo_ckpt))
        tok.save_pretrained(str(simpo_ckpt))
        m = {
            "checkpoint_name": "OCCAM-Qwen3-8B-SingleKing-SimPO", "num_pairs": len(pairs),
            "opt_steps": opt_step, "alpha_r": alpha_r, "eta_nll": eta_nll,
            "peak_vram_gb": round(peak_gb, 2), "wall_time_sec": round(wall_s, 2), "gpu_hours": round(wall_s / 3600.0, 4),
        }
        (simpo_ckpt / "training_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        del model, base, optimizer, scheduler
        cleanup_gpu()
        update_summary("qwen3_simpo_train", m)

    # Zero-Training Decoupled WiSE-FT on OCCAM-Qwen3-8B-SingleKing-SimPO (0 GPU sec!)
    sd = load_file(str(simpo_ckpt / "adapter_model.safetensors"))
    for tag, ga, gf in [("DecWiSE-a100-f35", 1.00, 0.35), ("DecWiSE-a90-f15", 0.90, 0.15)]:
        dst = CKPT_ROOT / f"OCCAM-Qwen3-8B-SimPO-{tag}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in simpo_ckpt.glob("*"):
            if f.is_file() and f.name != "adapter_model.safetensors":
                shutil.copy2(f, dst / f.name)
        scaled = {
            k: (v * ga if ("lora_B" in k and ".self_attn." in k) else (v * gf if ("lora_B" in k and ".mlp." in k) else v.clone()))
            for k, v in sd.items()
        }
        save_file(scaled, str(dst / "adapter_model.safetensors"))
        print(f"  [Qwen3 DecWiSE Ready] {dst.name} (attn={ga}, ffn_top={gf})")


# =========================================================================
# STAGE 5: SINGLE-ENGINE EVAL OF OCCAM-QWEN3-8B (SFT, SimPO, DecWiSE, SingleKing)
# =========================================================================
def run_q3_eval_occam(test_items: List[dict], hard_items: List[dict], min_free_gb: float = 20.5):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    gate = json.loads((TIER3_DATA / "qwen3_prefill_gate_preds.json").read_text(encoding="utf-8"))
    preds_8, preds_h = gate["preds_8bm"], gate["preds_hard"]

    adapters = [
        ("OCCAM-Qwen3-8B-SFT-Direct", CKPT_ROOT / "OCCAM-Qwen3-8B-LayerLoRA-SFT", 981),
        ("OCCAM-Qwen3-8B-SimPO-Direct", CKPT_ROOT / "OCCAM-Qwen3-8B-SingleKing-SimPO", 982),
        ("OCCAM-Qwen3-8B-SimPO-DecWiSE-f35-Direct", CKPT_ROOT / "OCCAM-Qwen3-8B-SimPO-DecWiSE-a100-f35", 983),
        ("OCCAM-Qwen3-8B-SimPO-DecWiSE-f15-Direct", CKPT_ROOT / "OCCAM-Qwen3-8B-SimPO-DecWiSE-a90-f15", 984),
    ]
    todo_8 = [(nm, ck, lid) for nm, ck, lid in adapters if not eval_mod.trace_file_complete(nm, len(test_items))]
    todo_h = [(nm, ck, lid) for nm, ck, lid in adapters if not (TRACES_DIR / f"HARD_{nm}.jsonl").exists()]

    if todo_8 or todo_h:
        print("\n" + "=" * 78)
        print("[TIER-3 STAGE 5] Single-Engine Qwen3-8B Multi-LoRA Eval across 4,526 + 268 Items")
        print("=" * 78)
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        util = safe_qwen3_vllm_util(min_free_gb=min_free_gb)
        llm = LLM(
            model=str(MODEL_DIR), trust_remote_code=True, dtype="bfloat16",
            max_model_len=4096, max_num_batched_tokens=1536, max_num_seqs=32,
            gpu_memory_utilization=util, enable_lora=True, max_loras=4, max_lora_rank=32,
            enable_prefix_caching=True, enforce_eager=True,
        )
        ps8 = [qwen3_prompt(tok, it["question"], thinking=False) for it in test_items]
        psh = [qwen3_prompt(tok, it["question"], thinking=False) for it in hard_items]

        for nm, ck, lid in todo_8:
            lr = LoRARequest(nm[:24], lid, str(ck))
            t0 = time.perf_counter()
            outs = llm.generate(ps8, SamplingParams(temperature=0.0, max_tokens=1024, stop=["</answer>"]), lora_request=lr)
            ws = time.perf_counter() - t0
            rows = []
            for it, o in zip(test_items, outs):
                txt = o.outputs[0].text
                if "<answer>" in txt and "</answer>" not in txt:
                    txt += "</answer>"
                ok, ext = grade_prediction(txt, it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
                rows.append({
                    "uid": it["uid"], "benchmark": it["benchmark"], "split_group": it["split_group"],
                    "ground_truth": it["ground_truth"], "extracted_answer": ext, "correct": ok,
                    "tokens": len(o.outputs[0].token_ids), "output": txt,
                    "wall_time_sec_total": round(ws, 2), "avg_batch_latency_ms": round(ws * 1000.0 / len(test_items), 2),
                })
            save_jsonl_safe(TRACES_DIR / f"{nm}.jsonl", rows)

        for nm, ck, lid in todo_h:
            lr = LoRARequest(nm[:24] + "h", lid + 10, str(ck))
            t0 = time.perf_counter()
            outs = llm.generate(psh, SamplingParams(temperature=0.0, max_tokens=1536, stop=["</answer>"]), lora_request=lr)
            ws = time.perf_counter() - t0
            rows = []
            for it, o in zip(hard_items, outs):
                txt = o.outputs[0].text
                if "<answer>" in txt and "</answer>" not in txt:
                    txt += "</answer>"
                ok, ext = grade_prediction(txt, it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
                rows.append({
                    "uid": it["uid"], "benchmark": it["benchmark"], "ground_truth": it["ground_truth"],
                    "extracted_answer": ext, "correct": ok, "tokens": len(o.outputs[0].token_ids),
                    "output": txt, "wall_time_sec_total": round(ws, 2),
                })
            save_jsonl_safe(TRACES_DIR / f"HARD_{nm}.jsonl", rows)

        del llm
        cleanup_gpu()

    # Compose TRUE SINGLE-BACKBONE SINGLE-ENGINE OCCAM-Qwen3-8B-SingleKing variants!
    # Framing -> LoRA ON (compact /no_think mental-model); Symbolic -> lora_request=None + /think ON (EXACT SAME 8B ENGINE!)
    res_8, res_h = {}, {}
    th8_map = {r["uid"]: r for r in load_jsonl_safe(TRACES_DIR / "Qwen3-8B-Think.jsonl")}
    thh_map = {r["uid"]: r for r in load_jsonl_safe(TRACES_DIR / "HARD_Qwen3-8B-Think.jsonl")}

    for nm, _, _ in adapters:
        res_8[nm] = summarize_8bm(nm)
        res_h[nm] = summarize_hard(nm)

        sk_nm = nm.replace("-Direct", "-SingleKing")
        f8_map = {r["uid"]: r for r in load_jsonl_safe(TRACES_DIR / f"{nm}.jsonl")}
        sk8_rows = []
        for it in test_items:
            reg = preds_8[it["uid"]]
            src = dict(th8_map[it["uid"]] if reg == "Symbolic_Rollout" else f8_map[it["uid"]])
            src["predicted_regime"] = reg
            sk8_rows.append(src)
        save_jsonl_safe(TRACES_DIR / f"{sk_nm}.jsonl", sk8_rows)
        res_8[sk_nm] = summarize_8bm(sk_nm)

        fh_map = {r["uid"]: r for r in load_jsonl_safe(TRACES_DIR / f"HARD_{nm}.jsonl")}
        skh_rows = []
        for it in hard_items:
            reg = preds_h[it["uid"]]
            src = dict(thh_map[it["uid"]] if reg == "Symbolic_Rollout" else fh_map[it["uid"]])
            src["predicted_regime"] = reg
            skh_rows.append(src)
        save_jsonl_safe(TRACES_DIR / f"HARD_{sk_nm}.jsonl", skh_rows)
        res_h[sk_nm] = summarize_hard(sk_nm)

    update_summary("qwen3_occam_singleking", {"8bm": res_8, "hard": res_h})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=["q3_prefill_gate", "q3_eval_native", "q3_sft_train", "q3_simpo_mine", "q3_simpo_train", "q3_eval_occam", "all_qwen3"],
    )
    parser.add_argument("--min-free-gb", type=float, default=20.2)
    args = parser.parse_args()

    test_items = load_jsonl_safe(SPLITS_DIR / "test_all_8benchmarks.jsonl")
    hard_items = load_jsonl_safe(HARD_PATH)

    if args.stage == "q3_prefill_gate":
        run_q3_prefill_gate(test_items, hard_items, min_free_gb=args.min_free_gb)
    elif args.stage == "q3_eval_native":
        run_q3_eval_native(test_items, hard_items, min_free_gb=args.min_free_gb)
    elif args.stage == "q3_sft_train":
        run_q3_sft_train(min_free_gb=args.min_free_gb)
    elif args.stage == "q3_simpo_mine":
        run_q3_simpo_mine(min_free_gb=args.min_free_gb)
    elif args.stage == "q3_simpo_train":
        run_q3_simpo_train(min_free_gb=args.min_free_gb)
    elif args.stage == "q3_eval_occam":
        run_q3_eval_occam(test_items, hard_items, min_free_gb=args.min_free_gb)
    elif args.stage == "all_qwen3":
        me = str(Path(__file__).resolve())
        stages = ["q3_prefill_gate", "q3_eval_native", "q3_sft_train", "q3_simpo_mine", "q3_simpo_train", "q3_eval_occam"]
        for i, st in enumerate(stages, 1):
            cmd = [sys.executable, me, "--stage", st, "--min-free-gb", str(args.min_free_gb)]
            print(f"\n{'#'*78}\n[TIER-3 QWEN3-8B ISOLATED STEP {i}/{len(stages)}] {st}\n{'#'*78}", flush=True)
            for attempt in range(1, 5):
                rc = subprocess.run(cmd).returncode
                if rc == 0:
                    break
                print(f"  [Retry {attempt}/4] {st} exited {rc}; waiting 60s...", flush=True)
                time.sleep(60)
            else:
                raise RuntimeError(f"Tier-3 stage {st} failed after 4 attempts")
        print("\n" + "=" * 78)
        print("[TIER-3 QWEN3-8B NUCLEAR STRIKE COMPLETE!] Summary ->", SUMMARY_PATH)
        print("=" * 78)


if __name__ == "__main__":
    main()
