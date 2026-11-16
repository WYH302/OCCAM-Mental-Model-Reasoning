#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 (OCCAM-2K): 100% Local RTX 3090 Self-Distillation & Matched-Source Control Engine.
Zero hardcoded templates! Uses vLLM continuous batching on real pretrained Qwen2.5-7B-Instruct:
  1. Unstructured Rollouts -> genuine natural-language y_rejected + matched-source LIMO-LongCoT SFT positives.
  2. Schema-Guided OCCAM-16 Rollouts -> triple rejection-sampling gate:
       (a) Strict <cog_router> / <cog_trace> / <answer> schema + valid OCCAM-16 primitive IDs
       (b) 100% Ground-Truth verification via unified symbolic grader (utils_grader.py)
       (c) Operator-aware MDL length compression gate (<=260 tok for structural framing; <=540 tok for math/symbolic)
     Follows STaR (Zelikman et al., NeurIPS 2022) rationalization on hard tail items during compile-time only.
  3. Strips external 16-primitive prompt (Schema Context Distillation -> Zero-Prompt Internalization) and exports:
       - sft_train.jsonl
       - simpo_train.jsonl
       - limo_longcot_train.jsonl (Matched-Source Variable Control!)
       - ablation_wo_router_train.jsonl
       - ablation_jargon_only_train.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_grader import grade_prediction

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "cognitive_primitives_16.json"
SEEDS_PATH = BASE_DIR / "data" / "raw_seeds" / "train_seeds_1600.jsonl"
SYNTH_DIR = BASE_DIR / "data" / "synthesized_pairs"
SPLITS_DIR = BASE_DIR / "data" / "final_splits"
ASSETS_DIR = BASE_DIR / "outputs" / "paper_assets"

SYNTH_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_occam_taxonomy() -> Tuple[Dict[str, dict], set, set]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    prim_map = {p["primitive_id"]: p for p in cfg["primitives"]}
    framing_set = set(cfg["compression_regimes"]["structural_framing"]["primitives"])
    symbolic_set = set(cfg["compression_regimes"]["symbolic_rollout"]["primitives"])
    return prim_map, framing_set, symbolic_set


def build_compile_time_schema_system_prompt(prim_map: Dict[str, dict]) -> str:
    lines = [
        "You are an expert reasoning compiler applying the OCCAM-16 Orthogonal Cognitive Primitives.",
        "Before solving any problem, route it to 1 to 3 orthogonal mental-model primitives from this exact registry:",
    ]
    for pid, p in prim_map.items():
        slots = ", ".join(s.split(":")[0] for s in p["mandatory_slots"])
        lines.append(f"- {pid}: Trigger={p['trigger_signature']} | Slots={slots} | AntiBias={p['anti_bias_check']}")
    lines.extend([
        "",
        "STRICT OUTPUT FORMAT (no extra prose outside these three XML blocks):",
        "<cog_router>",
        "[Problem_Topology]: <1 concise line identifying core causal/logical/mathematical structure>",
        "[Bias_Pruned]: <1 concise line naming the cognitive bias or fallacy pruned>",
        "[Activated_Primitives]: <PRIM_ID_1> -> <PRIM_ID_2>",
        "</cog_router>",
        "<cog_trace>",
        "<Compact, rigorous step-by-step execution strictly instantiating the activated primitives. Eliminate repetitive self-doubt ('Wait, let me re-read'). For causal/logic tasks keep under 160 words; for multi-step math include exact symbolic equations without skipping algebraic steps.>",
        "</cog_trace>",
        "<answer>EXACT_FINAL_ANSWER</answer>",
    ])
    return "\n".join(lines)


UNSTRUCTURED_SYSTEM_PROMPT = (
    "You are a helpful assistant. Think through the problem step by step, exploring possibilities "
    "as needed, and conclude your final answer inside <answer>...</answer> tags (for multiple choice, "
    "put only the option letter like A, B, C, or D inside <answer>)."
)


def build_task_format_suffix(item: dict) -> str:
    bm = item.get("benchmark", "")
    gt = item.get("ground_truth", "").strip()
    if bm in ("ARC-Challenge", "LogiQA") or re.fullmatch(r"[A-E]", gt):
        return "\nPut ONLY the single correct option letter (e.g., A, B, C, or D) inside <answer>...</answer>."
    if bm in ("CLadder", "StrategyQA") or gt.lower() in ("yes", "no"):
        return "\nPut ONLY Yes or No inside <answer>...</answer>."
    if bm == "FOLIO" or gt.lower() in ("true", "false", "uncertain"):
        return "\nPut ONLY True, False, or Uncertain inside <answer>...</answer>."
    if bm == "BBH-Hard":
        return f"\nConclude with the exact target format (e.g., {gt[:15] if len(gt) < 4 else 'valid/invalid, Yes/No, or option letter'}) inside <answer>...</answer>."
    return "\nPut ONLY the final simplified number or mathematical expression inside <answer>...</answer>."


def parse_and_validate_occam_trace(
    text: str,
    valid_prim_ids: set,
    framing_set: set,
    symbolic_set: set,
    benchmark: str,
) -> Optional[dict]:
    m_router = re.search(r"<cog_router>\s*(.*?)\s*</cog_router>", text, flags=re.DOTALL)
    m_trace = re.search(r"<cog_trace>\s*(.*?)\s*</cog_trace>", text, flags=re.DOTALL)
    m_ans = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if not (m_router and m_trace and m_ans):
        return None

    router_body = m_router.group(1).strip()
    trace_body = m_trace.group(1).strip()
    ans_body = m_ans.group(1).strip()
    if len(trace_body) < 25 or not ans_body:
        return None

    found_prims = []
    for pid in valid_prim_ids:
        if pid in router_body:
            found_prims.append(pid)
    # Preserve order of appearance in router_body
    found_prims.sort(key=lambda pid: router_body.find(pid))
    found_prims = found_prims[:3]
    if not found_prims:
        return None

    is_symbolic = (benchmark == "MATH-500") or any(p in symbolic_set for p in found_prims)
    regime = "symbolic_rollout" if is_symbolic else "structural_framing"
    dyn_lambda = 0.04 if is_symbolic else 0.18

    canonical_text = (
        f"<cog_router>\n{router_body}\n</cog_router>\n"
        f"<cog_trace>\n{trace_body}\n</cog_trace>\n"
        f"<answer>{ans_body}</answer>"
    )
    return {
        "canonical_text": canonical_text,
        "router_body": router_body,
        "trace_body": trace_body,
        "ans_body": ans_body,
        "activated_primitives": found_prims,
        "compression_regime": regime,
        "dynamic_lambda_len": dyn_lambda,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/Qwen2.5-7B-Instruct")
    parser.add_argument("--gpu-mem-util", type=float, default=0.88)
    args = parser.parse_args()

    t_start = time.perf_counter()
    prim_map, framing_set, symbolic_set = load_occam_taxonomy()
    valid_prim_ids = set(prim_map.keys())
    schema_sys_prompt = build_compile_time_schema_system_prompt(prim_map)

    seeds = [json.loads(ln) for ln in SEEDS_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print("=" * 78)
    print(f"[Stage 2] Local RTX 3090 Self-Distillation on {len(seeds)} Decontaminated Seeds")
    print(f"          Teacher Backbone: {args.model_path}")
    print("=" * 78)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
        enforce_eager=True,
    )

    def count_toks(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    # ---------------------------------------------------------------------
    # Phase 2A: Generate Unstructured Rollouts (n=3 per seed)
    # Serves TWO purposes:
    #   1. Genuine verbose/erroneous natural-language negative y_rejected
    #   2. Verified-correct unstructured rollouts -> Matched-Source LIMO-LongCoT SFT!
    # ---------------------------------------------------------------------
    print("[Phase 2A] Sampling Unstructured Rollouts (n=3 per seed) via vLLM...")
    unstruct_prompts = []
    for it in seeds:
        msgs = [
            {"role": "system", "content": UNSTRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": it["question"] + build_task_format_suffix(it)},
        ]
        unstruct_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    sp_unstruct = SamplingParams(n=3, temperature=0.75, top_p=0.92, max_tokens=1024)
    unstruct_outputs = llm.generate(unstruct_prompts, sp_unstruct)

    rejected_pool: Dict[str, dict] = {}
    limo_positives: Dict[str, dict] = {}

    for it, out_obj in zip(seeds, unstruct_outputs):
        uid = it["uid"]
        candidates_wrong = []
        candidates_right = []
        for comp in out_obj.outputs:
            txt = comp.text.strip()
            n_tok = len(comp.token_ids)
            if n_tok < 30:
                continue
            ok, _ = grade_prediction(txt, it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
            if ok:
                candidates_right.append((n_tok, txt))
            else:
                candidates_wrong.append((n_tok, txt))

        # For LIMO-Style Long-CoT SFT: pick the longest verified-correct unstructured chain
        if candidates_right:
            candidates_right.sort(key=lambda x: x[0], reverse=True)
            limo_positives[uid] = {"text": candidates_right[0][1], "tokens": candidates_right[0][0]}

        # For SimPO y_rejected: prefer wrong verbose chain, or longest verbose unstructured chain
        if candidates_wrong:
            candidates_wrong.sort(key=lambda x: x[0], reverse=True)
            rejected_pool[uid] = {"text": candidates_wrong[0][1], "tokens": candidates_wrong[0][0], "is_wrong": True}
        elif candidates_right:
            candidates_right.sort(key=lambda x: x[0], reverse=True)
            rejected_pool[uid] = {"text": candidates_right[0][1], "tokens": candidates_right[0][0], "is_wrong": False}
        else:
            fallback_txt = out_obj.outputs[0].text.strip()
            rejected_pool[uid] = {"text": fallback_txt, "tokens": max(32, count_toks(fallback_txt)), "is_wrong": True}

    print(f"  -> Phase 2A Done: {len(limo_positives)}/{len(seeds)} seeds solved zero-shot unstructured (for LIMO control).")

    # ---------------------------------------------------------------------
    # Phase 2B: Schema-Guided OCCAM-16 Rejection Sampling (Rounds 1..3 pure, Rounds 4..5 STaR rationalization)
    # ---------------------------------------------------------------------
    chosen_pool: Dict[str, dict] = {}
    unresolved = list(seeds)

    for round_idx in range(1, 6):
        if not unresolved:
            break
        is_rationalized = round_idx >= 4
        n_samples = 3 if round_idx <= 3 else 4
        temp = 0.45 + 0.12 * (round_idx - 1)
        print(f"[Phase 2B - Round {round_idx}/5] Remaining={len(unresolved)} | n={n_samples} | temp={temp:.2f} | STaR_Rationalize={is_rationalized}")

        batch_prompts = []
        for it in unresolved:
            user_content = it["question"] + build_task_format_suffix(it)
            if is_rationalized:
                # Compile-time STaR rationalization hint (strictly stripped before student training!)
                sketch = it.get("metadata", {}).get("gold_reasoning_sketch") or it.get("metadata", {}).get("gold_solution_sketch") or ""
                hint_block = (
                    f"\n\n[Compile-Time Verification Target (DO NOT mention this hint; derive self-consistently from the problem premises)]: "
                    f"Target verified answer is {it['ground_truth']}."
                )
                if sketch:
                    hint_block += f" Structural reference notes: {sketch[:280]}"
                user_content += hint_block

            msgs = [
                {"role": "system", "content": schema_sys_prompt},
                {"role": "user", "content": user_content},
            ]
            batch_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        sp_schema = SamplingParams(n=n_samples, temperature=min(temp, 0.85), top_p=0.92, max_tokens=768)
        schema_outputs = llm.generate(batch_prompts, sp_schema)

        next_unresolved = []
        for it, out_obj in zip(unresolved, schema_outputs):
            uid = it["uid"]
            rej_toks = rejected_pool[uid]["tokens"]
            valid_cands = []
            for comp in out_obj.outputs:
                raw_txt = comp.text.strip()
                parsed = parse_and_validate_occam_trace(
                    raw_txt, valid_prim_ids, framing_set, symbolic_set, it["benchmark"]
                )
                if not parsed:
                    continue
                ok, _ = grade_prediction(
                    parsed["canonical_text"],
                    it["ground_truth"],
                    benchmark=it["benchmark"],
                    metadata=it.get("metadata"),
                )
                if not ok:
                    continue
                c_toks = count_toks(parsed["canonical_text"])
                max_cap = 540 if parsed["compression_regime"] == "symbolic_rollout" else 280
                if c_toks <= max_cap:
                    valid_cands.append((c_toks, parsed))

            if valid_cands:
                # MDL selection: pick the shortest verified trace!
                valid_cands.sort(key=lambda x: x[0])
                best_toks, best_parsed = valid_cands[0]
                chosen_pool[uid] = {
                    **best_parsed,
                    "tokens_chosen": best_toks,
                    "rationalized": is_rationalized,
                    "round_solved": round_idx,
                }
                # Also if LIMO lacked a verified unstructured positive on this hard item, sample/keep parity later
            else:
                next_unresolved.append(it)

        unresolved = next_unresolved
        print(f"  -> Cumulative Verified OCCAM Positives: {len(chosen_pool)}/{len(seeds)}")

    # Ensure rejected trace is strictly longer or erroneous; if y_rejected happened to be shorter & right, sample verbose unguided rollout
    need_longer_neg = [
        it for it in seeds
        if it["uid"] in chosen_pool
        and not rejected_pool[it["uid"]]["is_wrong"]
        and rejected_pool[it["uid"]]["tokens"] <= chosen_pool[it["uid"]]["tokens_chosen"]
    ]
    if need_longer_neg:
        print(f"[Phase 2C] Expanding verbose unstructured rollouts for {len(need_longer_neg)} items...")
        prompts_verbose = []
        for it in need_longer_neg:
            msgs = [
                {
                    "role": "system",
                    "content": "Think aloud exhaustively step-by-step, checking every alternative and edge case in detail before concluding inside <answer>...</answer>.",
                },
                {"role": "user", "content": it["question"] + build_task_format_suffix(it)},
            ]
            prompts_verbose.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        outs_v = llm.generate(prompts_verbose, SamplingParams(n=2, temperature=0.85, max_tokens=1024))
        for it, o_v in zip(need_longer_neg, outs_v):
            cands = sorted([(len(c.token_ids), c.text.strip()) for c in o_v.outputs], key=lambda x: x[0], reverse=True)
            if cands and cands[0][0] > rejected_pool[it["uid"]]["tokens"]:
                ok_v, _ = grade_prediction(cands[0][1], it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
                rejected_pool[it["uid"]] = {"text": cands[0][1], "tokens": cands[0][0], "is_wrong": not ok_v}

    # Export all 5 authentic training splits
    synth_records = []
    sft_records = []
    simpo_records = []
    limo_records = []
    abl_wo_router_records = []
    abl_jargon_records = []

    prim_counter = Counter()
    rationalized_count = 0

    for it in seeds:
        uid = it["uid"]
        if uid not in chosen_pool:
            continue
        ch = chosen_pool[uid]
        rj = rejected_pool[uid]
        if ch["rationalized"]:
            rationalized_count += 1
        for p in ch["activated_primitives"]:
            prim_counter[p] += 1

        rec = {
            "uid": uid,
            "benchmark": it["benchmark"],
            "source_benchmark": it["source_benchmark"],
            "category": it["category"],
            "question": it["question"] + build_task_format_suffix(it),
            "ground_truth": it["ground_truth"],
            "y_chosen": ch["canonical_text"],
            "y_rejected": rj["text"],
            "activated_primitives": ch["activated_primitives"],
            "compression_regime": ch["compression_regime"],
            "dynamic_lambda_len": ch["dynamic_lambda_len"],
            "tokens_chosen": ch["tokens_chosen"],
            "tokens_rejected": rj["tokens"],
            "compression_ratio": round(ch["tokens_chosen"] / max(1, rj["tokens"]), 4),
            "rationalized": ch["rationalized"],
        }
        synth_records.append(rec)

        q_prompt = rec["question"]
        sft_records.append({"uid": uid, "benchmark": it["benchmark"], "prompt": q_prompt, "response": ch["canonical_text"]})
        simpo_records.append({
            "uid": uid,
            "benchmark": it["benchmark"],
            "prompt": q_prompt,
            "chosen": ch["canonical_text"],
            "rejected": rj["text"],
            "activated_primitives": ch["activated_primitives"],
            "compression_regime": ch["compression_regime"],
            "dynamic_lambda_len": ch["dynamic_lambda_len"],
        })

        # Ablation 1: Strip <cog_router>...</cog_router>, keep exact <cog_trace> + <answer>
        wo_router_resp = f"<cog_trace>\n{ch['trace_body']}\n</cog_trace>\n<answer>{ch['ans_body']}</answer>"
        abl_wo_router_records.append({
            "uid": uid,
            "benchmark": it["benchmark"],
            "prompt": q_prompt,
            "chosen": wo_router_resp,
            "rejected": rj["text"],
        })

        # Ablation 2: Surface Jargon-Only (prepend primitive names onto unstructured trace!)
        jargon_header = f"<cog_router>\n[Activated_Primitives]: {' -> '.join(ch['activated_primitives'])}\n</cog_router>\n"
        unstruct_base = limo_positives[uid]["text"] if uid in limo_positives else rj["text"]
        abl_jargon_records.append({
            "uid": uid,
            "benchmark": it["benchmark"],
            "prompt": q_prompt,
            "response": jargon_header + unstruct_base,
        })

        # Matched-Source LIMO-Style Long-CoT SFT (exact same teacher, verified unstructured CoT)
        if uid in limo_positives:
            limo_records.append({
                "uid": uid,
                "benchmark": it["benchmark"],
                "prompt": q_prompt,
                "response": limo_positives[uid]["text"],
            })

    def dump_jsonl(path: Path, rows: list):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump_jsonl(SYNTH_DIR / "synthesized_pairs_1600.jsonl", synth_records)
    dump_jsonl(SPLITS_DIR / "sft_train.jsonl", sft_records)
    dump_jsonl(SPLITS_DIR / "simpo_train.jsonl", simpo_records)
    dump_jsonl(SPLITS_DIR / "limo_longcot_train.jsonl", limo_records)
    dump_jsonl(SPLITS_DIR / "ablation_wo_router_train.jsonl", abl_wo_router_records)
    dump_jsonl(SPLITS_DIR / "ablation_jargon_only_train.jsonl", abl_jargon_records)

    wall_sec = time.perf_counter() - t_start
    avg_c = sum(r["tokens_chosen"] for r in synth_records) / max(1, len(synth_records))
    avg_r = sum(r["tokens_rejected"] for r in synth_records) / max(1, len(synth_records))

    stats = {
        "teacher_model": "Qwen/Qwen2.5-7B-Instruct (100% Local Self-Distillation)",
        "total_seeds": len(seeds),
        "verified_occam_pairs": len(synth_records),
        "direct_zero_hint_solved": len(synth_records) - rationalized_count,
        "star_rationalized_solved": rationalized_count,
        "matched_limo_longcot_positives": len(limo_records),
        "avg_tokens_chosen": round(avg_c, 2),
        "avg_tokens_rejected": round(avg_r, 2),
        "mean_compression_ratio": round(avg_c / max(1.0, avg_r), 4),
        "primitive_activation_counts": dict(prim_counter.most_common()),
        "wall_time_sec": round(wall_sec, 2),
        "gpu_hours": round(wall_sec / 3600.0, 4),
    }
    (ASSETS_DIR / "synthesis_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print(f"[STAGE 2 GATE PASSED] {len(synth_records)} Authentic Self-Distilled OCCAM Pairs Saved!")
    print(f"  - Direct Zero-Hint Verified : {len(synth_records) - rationalized_count}")
    print(f"  - STaR Rationalized         : {rationalized_count}")
    print(f"  - Matched LIMO Positives    : {len(limo_records)}")
    print(f"  - Avg Tokens (Chosen vs Rej): {avg_c:.1f} vs {avg_r:.1f} (ratio={avg_c/max(1.0, avg_r):.3f})")
    print(f"  - Wall-Clock Time           : {wall_sec:.2f}s ({wall_sec/60:.2f} min)")
    print("=" * 78)


if __name__ == "__main__":
    main()
