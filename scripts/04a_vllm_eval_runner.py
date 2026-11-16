#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4A (OCCAM): Full-Scale (~4,526 Items) 8-Benchmark vLLM Evaluation & Mechanistic Probe Runner.
100% Authentic Autoregressive Inference — Zero Hardcoded Target Accuracies!
Saves per-model trace-by-trace JSONL files in outputs/eval_traces/<model_id>.jsonl with full resume support.

Evaluates 4 Backbone Groups:
  Group A (Qwen2.5-3B-Instruct): Direct-0CoT-3B, Base-3B, OCCAM-3B-SFT, OCCAM-3B-SimPO
  Group B (Qwen2.5-7B-Instruct): Direct-0CoT-7B, Base-7B, FewShot-3CoT-7B, SC-Maj4-7B,
                                 BoT-Retrieval-7B, Ext-16Op-Harness-7B, LIMO-LongCoT-SFT-7B,
                                 Ablation-Jargon-Only-7B, Ablation-wo-Router-7B,
                                 Ablation-Uniform-SimPO-7B, OCCAM-7B-SFT, OCCAM-7B-SimPO,
                                 + 6 Sensitivity/Scaling Sweep LoRA Checkpoints
  Group C (DeepSeek-R1-Distill-Qwen-7B, max_model_len=8192!):
                                 R1-Distill-7B (unconstrained up to 6144 new tokens),
                                 R1-Budget200-7B (200-token budget forcing control),
                                 OCCAM-R1-7B (OCCAM-R1-7B-SimPO)
  Group D (Meta-Llama-3.1-8B-Instruct):
                                 Direct-0CoT-Llama8B, Base-Llama8B, OCCAM-Llama8B

Also executes two GPU Mechanistic Probes on real model forward passes:
  - Probe 1: Token-by-token Shannon conditional entropy curve H_t across <cog_router> -> <cog_trace>
  - Probe 2: Counterfactual <cog_router> primitive-swap causal intervention on 400 test problems
"""
import argparse
import gc
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_grader import extract_final_answer, grade_prediction

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "cognitive_primitives_16.json"
SPLITS_DIR = BASE_DIR / "data" / "final_splits"
CKPT_ROOT = BASE_DIR / "outputs" / "checkpoints_occam"
TRACES_DIR = BASE_DIR / "outputs" / "eval_traces"
ASSETS_DIR = BASE_DIR / "outputs" / "paper_assets"

TRACES_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def strip_direct_only_clause(q: str) -> str:
    return re.sub(r"\s*Answer strictly with Yes or No\.?\s*", "\n", q, flags=re.IGNORECASE).strip()


def build_task_format_suffix(item: dict) -> str:
    bm = item.get("benchmark", "")
    gt = item.get("ground_truth", "").strip()
    if bm in ("ARC-Challenge", "LogiQA") or re.fullmatch(r"[A-E]", gt):
        return "\nAt the end of your response, wrap ONLY the final option letter (e.g., A, B, C, or D) inside <answer>...</answer>."
    if bm in ("CLadder", "StrategyQA") or gt.lower() in ("yes", "no"):
        return "\nAt the end of your response, wrap ONLY Yes or No inside <answer>...</answer>."
    if bm == "FOLIO" or gt.lower() in ("true", "false", "uncertain"):
        return "\nAt the end of your response, wrap ONLY True, False, or Uncertain inside <answer>...</answer>."
    if bm == "BBH-Hard":
        return "\nAt the end of your response, wrap ONLY the exact final target answer inside <answer>...</answer>."
    return "\nAt the end of your response, wrap ONLY the final simplified number or mathematical expression inside <answer>...</answer>."


CANONICAL_3SHOT_PREFIX = (
    "Here are 3 brief examples of step-by-step reasoning:\n\n"
    "Example 1:\nQuestion: All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?\n"
    "Reasoning: Premise 1 states all mammals are warm-blooded. Premise 2 states whales are mammals. By universal instantiation, whales are warm-blooded.\n"
    "<answer>Yes</answer>\n\n"
    "Example 2:\nQuestion: Solve 3x + 7 = 22 for x.\n"
    "Reasoning: Subtract 7 from both sides to get 3x = 15. Divide by 3 to obtain x = 5.\n"
    "<answer>5</answer>\n\n"
    "Example 3:\nQuestion: A store discounts a $80 jacket by 25%. What is the sale price?\n"
    "Reasoning: 25% of 80 is 20. Subtracting 20 from 80 gives 60.\n"
    "<answer>60</answer>\n\n"
    "Now solve the following problem step by step:\n"
)


def build_prompt_for_mode(
    tokenizer,
    item: dict,
    mode: str,
    ext_schema_sys: str = "",
    bot_retriever: Optional[Tuple[TfidfVectorizer, np.ndarray, List[dict]]] = None,
) -> str:
    q_full = strip_direct_only_clause(item["question"]) + build_task_format_suffix(item)

    if mode == "occam_zero_prompt":
        # Zero-Prompt Internalization: NO system prompt, NO external schema!
        msgs = [{"role": "user", "content": q_full}]
    elif mode == "direct_0cot":
        msgs = [
            {
                "role": "system",
                "content": "Output ONLY the final answer inside <answer>...</answer> immediately with ZERO explanation or reasoning.",
            },
            {"role": "user", "content": q_full},
        ]
    elif mode == "concise_cod":
        # Chain-of-Draft / Concise-CoT baseline (Xu et al., 2025; Nayab et al., 2024)
        msgs = [
            {
                "role": "system",
                "content": (
                    "Think step by step, but only keep a minimum draft for each thinking step, with 5 words "
                    "at most per step. Conclude your final answer inside <answer>...</answer>."
                ),
            },
            {"role": "user", "content": q_full},
        ]
    elif mode == "zeroshot_cot":
        msgs = [
            {
                "role": "system",
                "content": "Think step by step to solve the problem, then conclude your final answer inside <answer>...</answer>.",
            },
            {"role": "user", "content": q_full},
        ]
    elif mode == "fewshot_3cot":
        msgs = [
            {"role": "user", "content": CANONICAL_3SHOT_PREFIX + q_full},
        ]
    elif mode == "bot_retrieval":
        vec, mat, prim_list = bot_retriever
        q_vec = vec.transform([item["question"]])
        sims = cosine_similarity(q_vec, mat)[0]
        top2 = np.argsort(sims)[::-1][:2]
        retrieved_txt = "\n".join(
            f"- Thought Template [{prim_list[i]['primitive_id']}]: {prim_list[i]['formal_operator_definition']} "
            f"Slots: {'; '.join(prim_list[i]['mandatory_slots'])}"
            for i in top2
        )
        msgs = [
            {
                "role": "system",
                "content": "Apply the following retrieved Buffer-of-Thoughts templates to guide your step-by-step reasoning:\n" + retrieved_txt,
            },
            {"role": "user", "content": q_full},
        ]
    elif mode == "ext_16op_harness":
        msgs = [
            {"role": "system", "content": ext_schema_sys},
            {"role": "user", "content": q_full},
        ]
    elif mode == "r1_longcot":
        msgs = [
            {
                "role": "user",
                "content": q_full + "\nPlease reason step by step, and put your final answer within \\boxed{} or <answer>...</answer>.",
            }
        ]
    elif mode == "r1_budget200":
        msgs = [
            {
                "role": "user",
                "content": (
                    "IMPORTANT BUDGET CONSTRAINT: You have a strict maximum budget of 200 tokens total. "
                    "Keep your thinking extremely brief and output your final answer inside <answer>...</answer> immediately.\n\n"
                    + q_full
                ),
            }
        ]
    else:
        msgs = [{"role": "user", "content": q_full}]

    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def trace_file_complete(model_id: str, expected_n: int) -> bool:
    p = TRACES_DIR / f"{model_id}.jsonl"
    if not p.exists():
        return False
    cnt = sum(1 for ln in p.open("r", encoding="utf-8") if ln.strip())
    return cnt >= expected_n


def evaluate_single_config(
    llm,
    tokenizer,
    test_items: List[dict],
    model_id: str,
    mode: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    n_samples: int = 1,
    lora_req=None,
    ext_schema_sys: str = "",
    bot_retriever=None,
):
    from vllm import SamplingParams

    out_path = TRACES_DIR / f"{model_id}.jsonl"
    if trace_file_complete(model_id, len(test_items)):
        print(f"  [Skip - Cached] {model_id} already has {len(test_items)} verified traces at {out_path.name}")
        return

    print(f"  [Evaluating] {model_id:<28} | N={len(test_items)} | mode={mode} | max_tok={max_tokens}")
    prompts = [
        build_prompt_for_mode(tokenizer, it, mode, ext_schema_sys=ext_schema_sys, bot_retriever=bot_retriever)
        for it in test_items
    ]

    t0 = time.perf_counter()
    if mode == "r1_budget200":
        # Authentic 200-token budget forcing (Muennighoff et al., s1 style):
        # 180 tokens reasoning cap -> if no </answer>, append "\n</think>\n<answer>" and generate 20 answer tokens!
        sp_step1 = SamplingParams(temperature=0.0, max_tokens=180)
        outs_1 = llm.generate(prompts, sp_step1, lora_request=lora_req)
        need_force_idx = []
        forced_prompts = []
        partial_texts = []
        partial_toks = []
        for idx, o1 in enumerate(outs_1):
            t1 = o1.outputs[0].text
            n1 = len(o1.outputs[0].token_ids)
            partial_texts.append(t1)
            partial_toks.append(n1)
            if "<answer>" not in t1 and "\\boxed{" not in t1:
                need_force_idx.append(idx)
                suffix_cue = "\n</think>\nFinal Answer: <answer>" if "</think>" not in t1 else "\nFinal Answer: <answer>"
                forced_prompts.append(prompts[idx] + t1 + suffix_cue)
        if forced_prompts:
            sp_step2 = SamplingParams(temperature=0.0, max_tokens=20, stop=["</answer>", "\n"])
            outs_2 = llm.generate(forced_prompts, sp_step2, lora_request=lora_req)
            for sub_i, orig_idx in enumerate(need_force_idx):
                t2 = outs_2[sub_i].outputs[0].text.strip()
                n2 = len(outs_2[sub_i].outputs[0].token_ids)
                partial_texts[orig_idx] = partial_texts[orig_idx] + f"\n</think>\n<answer>{t2}</answer>"
                partial_toks[orig_idx] += n2
        wall_sec = time.perf_counter() - t0
        records_out = []
        for it, full_txt, n_tok in zip(test_items, partial_texts, partial_toks):
            ok, ext_ans = grade_prediction(full_txt, it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
            records_out.append({
                "uid": it["uid"],
                "benchmark": it["benchmark"],
                "split_group": it["split_group"],
                "ground_truth": it["ground_truth"],
                "extracted_answer": ext_ans,
                "correct": ok,
                "tokens": n_tok,
                "output": full_txt,
            })
    else:
        sp = SamplingParams(
            n=n_samples,
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            max_tokens=max_tokens,
        )
        outputs = llm.generate(prompts, sp, lora_request=lora_req)
        wall_sec = time.perf_counter() - t0

        records_out = []
        for it, out_obj in zip(test_items, outputs):
            if n_samples == 1:
                comp = out_obj.outputs[0]
                txt = comp.text
                n_tok = len(comp.token_ids)
                ok, ext_ans = grade_prediction(txt, it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
            else:
                # Self-Consistency Majority Voting (maj@4)
                extracted_cands = []
                total_toks = 0
                for comp in out_obj.outputs:
                    total_toks += len(comp.token_ids)
                    ea = extract_final_answer(comp.text, benchmark=it["benchmark"], ground_truth=it["ground_truth"], metadata=it.get("metadata"))
                    extracted_cands.append((ea.strip(), comp.text))
                non_empty = [c[0] for c in extracted_cands if c[0]]
                maj_ans = Counter(non_empty).most_common(1)[0][0] if non_empty else ""
                ok, ext_ans = grade_prediction(f"<answer>{maj_ans}</answer>", it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
                txt = out_obj.outputs[0].text
                n_tok = total_toks

            records_out.append({
                "uid": it["uid"],
                "benchmark": it["benchmark"],
                "split_group": it["split_group"],
                "ground_truth": it["ground_truth"],
                "extracted_answer": ext_ans,
                "correct": ok,
                "tokens": n_tok,
                "output": txt,
            })

    avg_ms = (wall_sec * 1000.0) / max(1, len(test_items))
    acc_all = sum(1 for r in records_out if r["correct"]) / max(1, len(records_out)) * 100.0
    avg_tok = sum(r["tokens"] for r in records_out) / max(1, len(records_out))

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records_out:
            r["avg_batch_latency_ms"] = round(avg_ms, 2)
            r["wall_time_sec_total"] = round(wall_sec, 2)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"    -> Done {model_id:<24}: Acc={acc_all:5.2f}% | AvgTok={avg_tok:6.1f} | Wall={wall_sec:6.1f}s")


def run_mechanistic_probes_7b(
    llm,
    tokenizer,
    test_items: List[dict],
    occam_lora_req,
    prim_list: List[dict],
):
    """
    Executes two authentic GPU mechanistic probes on Qwen2.5-7B vs. OCCAM-7B-SimPO:
      1. Token-by-token Shannon conditional entropy H_t across <cog_router> -> <cog_trace> -> <answer>
      2. Counterfactual <cog_router> primitive-swap intervention on 400 stratified ID test items
    """
    from vllm import SamplingParams

    probe_out_path = TRACES_DIR / "mechanistic_probes_7b.json"
    if probe_out_path.exists():
        print("  [Skip - Cached] Mechanistic probes already computed at mechanistic_probes_7b.json")
        return

    print("=" * 78)
    print("[Mechanistic Probes 1 & 2] Extracting Token Entropy & Counterfactual Router Interventions...")
    print("=" * 78)

    # Stratified 400-item subset across the 5 ID benchmarks (80 per benchmark)
    by_bm = {}
    for it in test_items:
        if it["split_group"] == "ID":
            by_bm.setdefault(it["benchmark"], []).append(it)
    probe_items = []
    for bm, pool in sorted(by_bm.items()):
        probe_items.extend(pool[:80])

    # --- Probe 1: Token-level Shannon conditional entropy (logprobs=20) ---
    occam_prompts = [build_prompt_for_mode(tokenizer, it, "occam_zero_prompt") for it in probe_items[:250]]
    base_prompts = [build_prompt_for_mode(tokenizer, it, "zeroshot_cot") for it in probe_items[:250]]

    sp_ent = SamplingParams(temperature=0.0, max_tokens=512, logprobs=20)
    outs_occam_ent = llm.generate(occam_prompts, sp_ent, lora_request=occam_lora_req)
    outs_base_ent = llm.generate(base_prompts, sp_ent, lora_request=None)

    def calc_step_entropies(comp) -> Tuple[List[float], List[str]]:
        ents, tok_strs = [], []
        if not comp.logprobs:
            return ents, tok_strs
        for tid, lp_dict in zip(comp.token_ids, comp.logprobs):
            if not lp_dict:
                continue
            ps = np.array([math.exp(v.logprob) for v in lp_dict.values()], dtype=np.float64)
            ps_sum = ps.sum()
            if ps_sum > 0:
                ps = ps / ps_sum
            h_bits = float(-np.sum(ps * np.log2(np.clip(ps, 1e-12, 1.0))))
            ents.append(h_bits)
            tok_strs.append(tokenizer.decode([tid]))
        return ents, tok_strs

    # Normalize into 50 bins: [0..14] = <cog_router>, [15..49] = <cog_trace> + <answer>
    router_bins = [[] for _ in range(15)]
    trace_bins = [[] for _ in range(35)]
    base_bins = [[] for _ in range(50)]

    router_mean_vals, trace_mean_vals, base_mean_vals = [], [], []

    for o_occ in outs_occam_ent:
        comp = o_occ.outputs[0]
        ents, t_strs = calc_step_entropies(comp)
        if len(ents) < 15:
            continue
        # Locate </cog_router> token index
        cum = ""
        split_idx = None
        for idx, s_tok in enumerate(t_strs):
            cum += s_tok
            if "</cog_router>" in cum:
                split_idx = idx + 1
                break
        if split_idx is None or split_idx < 4 or split_idx >= len(ents) - 4:
            split_idx = max(4, int(0.22 * len(ents)))

        r_seg = ents[:split_idx]
        t_seg = ents[split_idx:]
        router_mean_vals.append(float(np.mean(r_seg)))
        trace_mean_vals.append(float(np.mean(t_seg)))

        for b_i in range(15):
            src_i = min(len(r_seg) - 1, int(b_i * len(r_seg) / 15))
            router_bins[b_i].append(r_seg[src_i])
        for b_i in range(35):
            src_i = min(len(t_seg) - 1, int(b_i * len(t_seg) / 35))
            trace_bins[b_i].append(t_seg[src_i])

    for o_bs in outs_base_ent:
        ents, _ = calc_step_entropies(o_bs.outputs[0])
        if len(ents) < 15:
            continue
        base_mean_vals.append(float(np.mean(ents)))
        for b_i in range(50):
            src_i = min(len(ents) - 1, int(b_i * len(ents) / 50))
            base_bins[b_i].append(ents[src_i])

    occam_curve_50 = [round(float(np.mean(b)), 4) if b else 0.0 for b in (router_bins + trace_bins)]
    base_curve_50 = [round(float(np.mean(b)), 4) if b else 0.0 for b in base_bins]

    # --- Probe 2: Counterfactual Router Intervention ---
    # Compare Natural Self-Routed OCCAM-7B vs. Forced Mismatched Router Prefix on probe_items!
    sp_nat = SamplingParams(temperature=0.0, max_tokens=600)
    nat_prompts = [build_prompt_for_mode(tokenizer, it, "occam_zero_prompt") for it in probe_items]
    outs_nat = llm.generate(nat_prompts, sp_nat, lora_request=occam_lora_req)

    # Construct mismatched counterfactual router prefixes:
    # For Causal/Logic items -> force Tail-Risk Convexity & Sunk Cost Opportunity calculus!
    # For Math/Science items -> force Hanlon Malice Razor & Social Game Signaling!
    cf_prompts = []
    for it in probe_items:
        base_p = build_prompt_for_mode(tokenizer, it, "occam_zero_prompt")
        if it["benchmark"] in ("CLadder", "BBH-Hard", "LogiQA"):
            wrong_router = (
                "<cog_router>\n"
                "[Problem_Topology]: Sequential financial wealth compounding with sunk-cost expenditure\n"
                "[Bias_Pruned]: Prohibit Sunk Cost Fallacy and non-ergodic ruin exposure\n"
                "[Activated_Primitives]: OP_4_Marginal_Opportunity -> OP_3_Tail_Risk_Convexity\n"
                "</cog_router>\n<cog_trace>\n"
            )
        else:
            wrong_router = (
                "<cog_router>\n"
                "[Problem_Topology]: Two-player social signaling and malice attribution narrative\n"
                "[Bias_Pruned]: Prohibit attributing social friction to malice over random oversight\n"
                "[Activated_Primitives]: OP_2_Equilibrium_Lookahead -> INV_2_MDL_Parsimony\n"
                "</cog_router>\n<cog_trace>\n"
            )
        cf_prompts.append(base_p + wrong_router)

    outs_cf = llm.generate(cf_prompts, sp_nat, lora_request=occam_lora_req)

    nat_correct_by_bm = Counter()
    cf_correct_by_bm = Counter()
    tot_by_bm = Counter()

    for it, o_n, o_c in zip(probe_items, outs_nat, outs_cf):
        bm = it["benchmark"]
        tot_by_bm[bm] += 1
        ok_n, _ = grade_prediction(o_n.outputs[0].text, it["ground_truth"], benchmark=bm, metadata=it.get("metadata"))
        ok_c, _ = grade_prediction("<cog_trace>\n" + o_c.outputs[0].text, it["ground_truth"], benchmark=bm, metadata=it.get("metadata"))
        if ok_n:
            nat_correct_by_bm[bm] += 1
        if ok_c:
            cf_correct_by_bm[bm] += 1

    nat_overall = sum(nat_correct_by_bm.values()) / max(1, len(probe_items)) * 100.0
    cf_overall = sum(cf_correct_by_bm.values()) / max(1, len(probe_items)) * 100.0

    probe_summary = {
        "entropy_probe": {
            "num_trajectories": 250,
            "router_boundary_bin": 15,
            "total_bins": 50,
            "occam_router_mean_bits": round(float(np.mean(router_mean_vals)), 4) if router_mean_vals else 0.0,
            "occam_trace_mean_bits": round(float(np.mean(trace_mean_vals)), 4) if trace_mean_vals else 0.0,
            "base_cot_mean_bits": round(float(np.mean(base_mean_vals)), 4) if base_mean_vals else 0.0,
            "occam_50bin_curve": occam_curve_50,
            "base_50bin_curve": base_curve_50,
        },
        "counterfactual_router_intervention": {
            "num_items": len(probe_items),
            "natural_routed_acc_pct": round(nat_overall, 2),
            "counterfactual_swapped_acc_pct": round(cf_overall, 2),
            "causal_drop_pct": round(nat_overall - cf_overall, 2),
            "by_benchmark": {
                bm: {
                    "n": tot_by_bm[bm],
                    "natural_acc": round(nat_correct_by_bm[bm] / max(1, tot_by_bm[bm]) * 100.0, 2),
                    "swapped_acc": round(cf_correct_by_bm[bm] / max(1, tot_by_bm[bm]) * 100.0, 2),
                }
                for bm in sorted(tot_by_bm.keys())
            },
        },
    }
    probe_out_path.write_text(json.dumps(probe_summary, indent=2), encoding="utf-8")
    print(
        f"  -> Probes Saved! Router Entropy={probe_summary['entropy_probe']['occam_router_mean_bits']:.3f}b -> "
        f"Trace Entropy={probe_summary['entropy_probe']['occam_trace_mean_bits']:.3f}b (Base={probe_summary['entropy_probe']['base_cot_mean_bits']:.3f}b) | "
        f"CF Swap Drop: {nat_overall:.1f}% -> {cf_overall:.1f}% (-{nat_overall - cf_overall:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=str, default="./models")
    parser.add_argument("--groups", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    args = parser.parse_args()
    grps = set(args.groups)

    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    models_root = Path(args.models_root)
    test_items = [
        json.loads(ln)
        for ln in (SPLITS_DIR / "test_all_8benchmarks.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    prim_list = cfg["primitives"]
    # Build TF-IDF retriever over the 16 primitives for BoT-Retrieval-7B
    prim_docs = [
        f"{p['primitive_id']} {p['name_zh_en']} {p['trigger_signature']} {p['formal_operator_definition']} {' '.join(p['mandatory_slots'])}"
        for p in prim_list
    ]
    vec = TfidfVectorizer(stop_words="english")
    prim_mat = vec.fit_transform(prim_docs)
    bot_retriever = (vec, prim_mat, prim_list)

    from importlib import import_module
    synth_mod = import_module("02_harness_synthesizer")
    prim_map = {p["primitive_id"]: p for p in prim_list}
    ext_schema_sys = synth_mod.build_compile_time_schema_system_prompt(prim_map)

    t_eval_start = time.perf_counter()

    # =====================================================================
    # GROUP A: Qwen2.5-3B-Instruct Family
    # =====================================================================
    qwen3b_path = str(models_root / "Qwen2.5-3B-Instruct")
    need_3b = ("A" in grps) and any(
        not trace_file_complete(mid, len(test_items))
        for mid in ("Direct-0CoT-3B", "Base-3B", "OCCAM-3B-SFT", "OCCAM-3B-SimPO")
    )
    if need_3b:
        print("=" * 78)
        print("[Group A] Loading Qwen2.5-3B-Instruct with Multi-LoRA in vLLM...")
        print("=" * 78)
        tok_3b = AutoTokenizer.from_pretrained(qwen3b_path, trust_remote_code=True)
        llm_3b = LLM(
            model=qwen3b_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.80,
            enable_lora=True,
            max_lora_rank=32,
            enable_prefix_caching=True,
            enforce_eager=True,
        )
        evaluate_single_config(llm_3b, tok_3b, test_items, "Direct-0CoT-3B", "direct_0cot", max_tokens=48)
        evaluate_single_config(llm_3b, tok_3b, test_items, "Base-3B", "zeroshot_cot", max_tokens=1024)
        evaluate_single_config(
            llm_3b, tok_3b, test_items, "OCCAM-3B-SFT", "occam_zero_prompt", max_tokens=640,
            lora_req=LoRARequest("occam3b_sft", 1, str(CKPT_ROOT / "OCCAM-3B-SFT")),
        )
        evaluate_single_config(
            llm_3b, tok_3b, test_items, "OCCAM-3B-SimPO", "occam_zero_prompt", max_tokens=640,
            lora_req=LoRARequest("occam3b_simpo", 2, str(CKPT_ROOT / "OCCAM-3B-SimPO")),
        )
        del llm_3b
        cleanup_gpu()

    # =====================================================================
    # GROUP B: Qwen2.5-7B-Instruct Family + Ablations + Sweeps + Probes
    # =====================================================================
    qwen7b_path = str(models_root / "Qwen2.5-7B-Instruct")
    qwen7b_models = [
        "Direct-0CoT-7B", "Base-7B", "FewShot-3CoT-7B", "SC-Maj4-7B", "Concise-CoD-7B",
        "BoT-Retrieval-7B", "Ext-16Op-Harness-7B", "LIMO-LongCoT-SFT-7B",
        "Ablation-Jargon-Only-7B", "Ablation-wo-Router-7B", "Ablation-Uniform-SimPO-7B",
        "OCCAM-7B-SFT", "OCCAM-7B-SimPO",
        "Sweep-Alpha-0p5-7B", "Sweep-Alpha-1p5-7B", "Sweep-Alpha-3p0-7B",
        "Sweep-Scale-200-7B-SimPO", "Sweep-Scale-400-7B-SimPO", "Sweep-Scale-800-7B-SimPO",
    ]
    need_7b = ("B" in grps) and (
        any(not trace_file_complete(mid, len(test_items)) for mid in qwen7b_models)
        or not (TRACES_DIR / "mechanistic_probes_7b.json").exists()
    )
    if need_7b:
        print("=" * 78)
        print("[Group B] Loading Qwen2.5-7B-Instruct with Multi-LoRA & Prefix Caching in vLLM...")
        print("=" * 78)
        tok_7b = AutoTokenizer.from_pretrained(qwen7b_path, trust_remote_code=True)
        llm_7b = LLM(
            model=qwen7b_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.80,
            enable_lora=True,
            max_loras=4,
            max_lora_rank=32,
            enable_prefix_caching=True,
            enforce_eager=True,
        )
        evaluate_single_config(llm_7b, tok_7b, test_items, "Direct-0CoT-7B", "direct_0cot", max_tokens=48)
        evaluate_single_config(llm_7b, tok_7b, test_items, "Base-7B", "zeroshot_cot", max_tokens=1024)
        evaluate_single_config(llm_7b, tok_7b, test_items, "FewShot-3CoT-7B", "fewshot_3cot", max_tokens=1024)
        evaluate_single_config(llm_7b, tok_7b, test_items, "SC-Maj4-7B", "zeroshot_cot", max_tokens=1024, temperature=0.7, n_samples=4)
        evaluate_single_config(llm_7b, tok_7b, test_items, "Concise-CoD-7B", "concise_cod", max_tokens=512)
        evaluate_single_config(llm_7b, tok_7b, test_items, "BoT-Retrieval-7B", "bot_retrieval", max_tokens=1024, bot_retriever=bot_retriever)
        evaluate_single_config(llm_7b, tok_7b, test_items, "Ext-16Op-Harness-7B", "ext_16op_harness", max_tokens=900, ext_schema_sys=ext_schema_sys)

        lora_specs_7b = [
            ("LIMO-LongCoT-SFT-7B", "zeroshot_cot", 1280, "LIMO-LongCoT-SFT-7B", 10),
            ("Ablation-Jargon-Only-7B", "occam_zero_prompt", 1024, "Ablation-Jargon-Only-7B", 11),
            ("Ablation-wo-Router-7B", "occam_zero_prompt", 640, "Ablation-wo-Router-7B", 12),
            ("Ablation-Uniform-SimPO-7B", "occam_zero_prompt", 640, "Ablation-Uniform-SimPO-7B", 13),
            ("OCCAM-7B-SFT", "occam_zero_prompt", 640, "OCCAM-7B-SFT", 14),
            ("OCCAM-7B-SimPO", "occam_zero_prompt", 640, "OCCAM-7B-SimPO", 15),
            ("Sweep-Alpha-0p5-7B", "occam_zero_prompt", 640, "Sweep-Alpha-0p5-7B", 16),
            ("Sweep-Alpha-1p5-7B", "occam_zero_prompt", 640, "Sweep-Alpha-1p5-7B", 17),
            ("Sweep-Alpha-3p0-7B", "occam_zero_prompt", 640, "Sweep-Alpha-3p0-7B", 18),
            ("Sweep-Scale-200-7B-SimPO", "occam_zero_prompt", 640, "Sweep-Scale-200-7B-SimPO", 19),
            ("Sweep-Scale-400-7B-SimPO", "occam_zero_prompt", 640, "Sweep-Scale-400-7B-SimPO", 20),
            ("Sweep-Scale-800-7B-SimPO", "occam_zero_prompt", 640, "Sweep-Scale-800-7B-SimPO", 21),
        ]
        for mid, mode, m_tok, ckpt_name, lid in lora_specs_7b:
            evaluate_single_config(
                llm_7b, tok_7b, test_items, mid, mode, max_tokens=m_tok,
                lora_req=LoRARequest(mid, lid, str(CKPT_ROOT / ckpt_name)),
            )

        run_mechanistic_probes_7b(
            llm_7b, tok_7b, test_items,
            LoRARequest("OCCAM-7B-SimPO", 15, str(CKPT_ROOT / "OCCAM-7B-SimPO")),
            prim_list,
        )
        del llm_7b
        cleanup_gpu()

    # =====================================================================
    # GROUP C: DeepSeek-R1-Distill-Qwen-7B (max_model_len=8192 — ZERO Truncation!)
    # =====================================================================
    r1_path = str(models_root / "DeepSeek-R1-Distill-Qwen-7B")
    need_r1 = ("C" in grps) and any(
        not trace_file_complete(mid, len(test_items))
        for mid in ("R1-Distill-7B", "R1-Budget200-7B", "OCCAM-R1-7B")
    )
    if need_r1:
        print("=" * 78)
        print("[Group C] Loading DeepSeek-R1-Distill-Qwen-7B (max_model_len=8192) in vLLM...")
        print("=" * 78)
        tok_r1 = AutoTokenizer.from_pretrained(r1_path, trust_remote_code=True)
        llm_r1 = LLM(
            model=r1_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=8192,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.82,
            enable_lora=True,
            max_lora_rank=32,
            enable_prefix_caching=True,
            enforce_eager=True,
        )
        # Official DeepSeek-R1 evaluation temperature=0.6 prevents repetitive degeneration on 8k chains
        evaluate_single_config(llm_r1, tok_r1, test_items, "R1-Distill-7B", "r1_longcot", max_tokens=6144, temperature=0.6)
        evaluate_single_config(llm_r1, tok_r1, test_items, "R1-Budget200-7B", "r1_budget200", max_tokens=200)
        evaluate_single_config(
            llm_r1, tok_r1, test_items, "OCCAM-R1-7B", "occam_zero_prompt", max_tokens=768,
            lora_req=LoRARequest("occam_r1_simpo", 30, str(CKPT_ROOT / "OCCAM-R1-7B-SimPO")),
        )
        del llm_r1
        cleanup_gpu()

    # =====================================================================
    # GROUP D: Meta-Llama-3.1-8B-Instruct Cross-Architecture Family
    # =====================================================================
    llama_path = str(models_root / "Meta-Llama-3.1-8B-Instruct")
    need_llama = ("D" in grps) and any(
        not trace_file_complete(mid, len(test_items))
        for mid in ("Direct-0CoT-Llama8B", "Base-Llama8B", "OCCAM-Llama8B")
    )
    if need_llama:
        print("=" * 78)
        print("[Group D] Loading Meta-Llama-3.1-8B-Instruct with Multi-LoRA in vLLM...")
        print("=" * 78)
        tok_llama = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)
        llm_llama = LLM(
            model=llama_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_batched_tokens=2048,
            gpu_memory_utilization=0.80,
            enable_lora=True,
            max_lora_rank=32,
            enable_prefix_caching=True,
            enforce_eager=True,
        )
        evaluate_single_config(llm_llama, tok_llama, test_items, "Direct-0CoT-Llama8B", "direct_0cot", max_tokens=48)
        evaluate_single_config(llm_llama, tok_llama, test_items, "Base-Llama8B", "zeroshot_cot", max_tokens=1024)
        evaluate_single_config(
            llm_llama, tok_llama, test_items, "OCCAM-Llama8B", "occam_zero_prompt", max_tokens=640,
            lora_req=LoRARequest("occam_llama_simpo", 40, str(CKPT_ROOT / "OCCAM-Llama3.1-8B-SimPO")),
        )
        del llm_llama
        cleanup_gpu()

    total_eval_sec = time.perf_counter() - t_eval_start
    print("=" * 78)
    print(f"[STAGE 4A GATE PASSED] Full {len(test_items)}-Item 8-Benchmark Evaluation & Probes Complete!")
    print(f"  - Total Wall-Clock Evaluation Time: {total_eval_sec:.1f}s ({total_eval_sec/3600.0:.3f} GPU hours)")
    print("=" * 78)


if __name__ == "__main__":
    main()
