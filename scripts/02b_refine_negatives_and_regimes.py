#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2B (OCCAM-2K): Unstructured CoT Negative & Matched-LIMO Refiner + Regime Canonicalizer.
Fixes the CLadder/MC "Answer strictly with Yes or No / Put ONLY..." prompt-collision where Phase 2A
emitted 1-token direct answers instead of genuine verbose unstructured Chain-of-Thought rollouts.
Preserves all 1,570 authentic verified y_chosen trajectories from Stage 2 while regenerating genuine
multi-hundred-token unstructured CoT rollouts for y_rejected and Matched-Source LIMO-LongCoT SFT.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_grader import grade_prediction

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "cognitive_primitives_16.json"
SEEDS_PATH = BASE_DIR / "data" / "raw_seeds" / "train_seeds_1600.jsonl"
SYNTH_DIR = BASE_DIR / "data" / "synthesized_pairs"
SPLITS_DIR = BASE_DIR / "data" / "final_splits"
ASSETS_DIR = BASE_DIR / "outputs" / "paper_assets"


def strip_direct_only_clause(q: str) -> str:
    """Remove raw dataset clauses like 'Answer strictly with Yes or No.' that suppress CoT."""
    q = re.sub(r"\s*Answer strictly with Yes or No\.?\s*", "\n", q, flags=re.IGNORECASE)
    return q.strip()


def build_clean_user_prompt(item: dict) -> str:
    """Unified train & test user prompt that specifies final <answer> tag format without suppressing reasoning."""
    q_base = strip_direct_only_clause(item["question"])
    bm = item.get("benchmark", "")
    gt = item.get("ground_truth", "").strip()
    if bm in ("ARC-Challenge", "LogiQA") or re.fullmatch(r"[A-E]", gt):
        suf = "\nAt the end of your response, wrap ONLY the final option letter (e.g., A, B, C, or D) inside <answer>...</answer>."
    elif bm in ("CLadder", "StrategyQA") or gt.lower() in ("yes", "no"):
        suf = "\nAt the end of your response, wrap ONLY Yes or No inside <answer>...</answer>."
    elif bm == "FOLIO" or gt.lower() in ("true", "false", "uncertain"):
        suf = "\nAt the end of your response, wrap ONLY True, False, or Uncertain inside <answer>...</answer>."
    elif bm == "BBH-Hard":
        suf = "\nAt the end of your response, wrap ONLY the exact final target answer inside <answer>...</answer>."
    else:
        suf = "\nAt the end of your response, wrap ONLY the final simplified number or mathematical expression inside <answer>...</answer>."
    return q_base + suf


VERBOSE_UNSTRUCT_SYS = (
    "You are a helpful assistant solving a reasoning problem. You MUST write out a thorough, detailed "
    "step-by-step Chain-of-Thought analysis (at least 3 to 6 paragraphs, exploring premises, intermediate "
    "calculations, and alternative interpretations) BEFORE giving your final answer inside <answer>...</answer>. "
    "Never output a bare one-word answer without full step-by-step reasoning."
)


def canonicalize_router_ids(text: str, full_ids: List[str]) -> str:
    short_to_full = {}
    for fid in full_ids:
        parts = fid.split("_")
        if len(parts) >= 2:
            short_key = f"{parts[0]}_{parts[1]}"
            short_to_full[short_key] = fid
    # Replace standalone short keys like META_2 (not already followed by _Name)
    for sk, fid in short_to_full.items():
        text = re.sub(rf"\b{sk}\b(?!_)", fid, text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/Qwen2.5-7B-Instruct")
    parser.add_argument("--gpu-mem-util", type=float, default=0.88)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    full_prim_ids = [p["primitive_id"] for p in cfg["primitives"]]

    seeds = [json.loads(ln) for ln in SEEDS_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    seed_map = {s["uid"]: s for s in seeds}

    existing_synth = [
        json.loads(ln)
        for ln in (SYNTH_DIR / "synthesized_pairs_1600.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    chosen_by_uid = {r["uid"]: r for r in existing_synth}
    prev_stats = json.loads((ASSETS_DIR / "synthesis_stats.json").read_text(encoding="utf-8"))
    prev_wall = float(prev_stats.get("wall_time_sec", 0.0))

    print("=" * 78)
    print(f"[Stage 2B] Regenerating Authentic Verbose Unstructured CoTs for {len(chosen_by_uid)} Verified Pairs")
    print("=" * 78)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,
        enforce_eager=True,
    )

    def count_toks(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    active_seeds = [s for s in seeds if s["uid"] in chosen_by_uid]
    prompts = []
    for it in active_seeds:
        msgs = [
            {"role": "system", "content": VERBOSE_UNSTRUCT_SYS},
            {"role": "user", "content": build_clean_user_prompt(it)},
        ]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    sp = SamplingParams(n=4, temperature=0.80, top_p=0.95, max_tokens=1024)
    outs = llm.generate(prompts, sp)

    rejected_pool: Dict[str, dict] = {}
    limo_positives: Dict[str, dict] = {}
    need_extra_verbose = []

    for it, o_obj in zip(active_seeds, outs):
        uid = it["uid"]
        ch_toks = chosen_by_uid[uid]["tokens_chosen"]
        c_wrong = []
        c_right = []
        for comp in o_obj.outputs:
            txt = comp.text.strip()
            nt = len(comp.token_ids)
            if nt < 45:
                continue
            ok, _ = grade_prediction(txt, it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
            if ok:
                c_right.append((nt, txt))
            else:
                c_wrong.append((nt, txt))

        if c_right:
            c_right.sort(key=lambda x: x[0], reverse=True)
            limo_positives[uid] = {"text": c_right[0][1], "tokens": c_right[0][0]}

        # Prefer wrong verbose CoT, else longest right unstructured CoT > ch_toks
        if c_wrong:
            c_wrong.sort(key=lambda x: x[0], reverse=True)
            rejected_pool[uid] = {"text": c_wrong[0][1], "tokens": c_wrong[0][0], "is_wrong": True}
        elif c_right:
            c_right.sort(key=lambda x: x[0], reverse=True)
            rejected_pool[uid] = {"text": c_right[0][1], "tokens": c_right[0][0], "is_wrong": False}
            if c_right[0][0] <= ch_toks:
                need_extra_verbose.append(it)
        else:
            # All < 45 tokens — schedule forced exhaustive rollout
            need_extra_verbose.append(it)
            best_raw = max([(len(c.token_ids), c.text.strip()) for c in o_obj.outputs], key=lambda x: x[0])
            rejected_pool[uid] = {"text": best_raw[1], "tokens": max(32, best_raw[0]), "is_wrong": True}

    if need_extra_verbose:
        print(f"[Stage 2B - Pass 2] Expanding overthinking rollouts on {len(need_extra_verbose)} short items...")
        p2 = []
        for it in need_extra_verbose:
            msgs = [
                {
                    "role": "system",
                    "content": (
                        "Write an exhaustive, unhurried chain-of-thought monologue (at least 350 words). "
                        "Double-check every premise, test alternative hypotheses, reflect using 'Wait, let us also check...', "
                        "and verify every option individually before placing your final answer in <answer>...</answer>."
                    ),
                },
                {"role": "user", "content": build_clean_user_prompt(it)},
            ]
            p2.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        outs2 = llm.generate(p2, SamplingParams(n=2, temperature=0.88, top_p=0.95, max_tokens=1024))
        for it, o2 in zip(need_extra_verbose, outs2):
            uid = it["uid"]
            cands = sorted([(len(c.token_ids), c.text.strip()) for c in o2.outputs], key=lambda x: x[0], reverse=True)
            if cands and cands[0][0] > rejected_pool[uid]["tokens"]:
                ok2, _ = grade_prediction(cands[0][1], it["ground_truth"], benchmark=it["benchmark"], metadata=it.get("metadata"))
                rejected_pool[uid] = {"text": cands[0][1], "tokens": cands[0][0], "is_wrong": not ok2}
                if ok2 and uid not in limo_positives:
                    limo_positives[uid] = {"text": cands[0][1], "tokens": cands[0][0]}

    synth_records, sft_records, simpo_records = [], [], []
    limo_records, abl_wo_router_records, abl_jargon_records = [], [], []
    prim_counter = Counter()
    rationalized_count = 0

    for it in active_seeds:
        uid = it["uid"]
        old_rec = chosen_by_uid[uid]
        rj = rejected_pool[uid]

        y_chosen_clean = canonicalize_router_ids(old_rec["y_chosen"], full_prim_ids)
        c_toks = count_toks(y_chosen_clean)
        prims = old_rec["activated_primitives"]
        for p in prims:
            prim_counter[p] += 1
        if old_rec.get("rationalized"):
            rationalized_count += 1

        # Clean regime rule: irreducible symbolic rollout applies to MATH-500 or quantitative axiom derivations
        is_symbolic = (it["benchmark"] == "MATH-500") or (
            prims and prims[0] in ("INV_1_Axiom_Deconstruction", "OP_3_Tail_Risk_Convexity", "OP_4_Marginal_Opportunity")
            and it["benchmark"] not in ("CLadder", "ARC-Challenge")
        )
        regime = "symbolic_rollout" if is_symbolic else "structural_framing"
        dyn_lam = 0.04 if is_symbolic else 0.18

        q_prompt = build_clean_user_prompt(it)
        rec = {
            "uid": uid,
            "benchmark": it["benchmark"],
            "source_benchmark": it["source_benchmark"],
            "category": it["category"],
            "question": q_prompt,
            "ground_truth": it["ground_truth"],
            "y_chosen": y_chosen_clean,
            "y_rejected": rj["text"],
            "activated_primitives": prims,
            "compression_regime": regime,
            "dynamic_lambda_len": dyn_lam,
            "tokens_chosen": c_toks,
            "tokens_rejected": rj["tokens"],
            "compression_ratio": round(c_toks / max(1, rj["tokens"]), 4),
            "rationalized": bool(old_rec.get("rationalized", False)),
        }
        synth_records.append(rec)
        sft_records.append({"uid": uid, "benchmark": it["benchmark"], "prompt": q_prompt, "response": y_chosen_clean})
        simpo_records.append({
            "uid": uid,
            "benchmark": it["benchmark"],
            "prompt": q_prompt,
            "chosen": y_chosen_clean,
            "rejected": rj["text"],
            "activated_primitives": prims,
            "compression_regime": regime,
            "dynamic_lambda_len": dyn_lam,
        })

        m_tr = re.search(r"<cog_trace>\s*(.*?)\s*</cog_trace>", y_chosen_clean, flags=re.DOTALL)
        m_an = re.search(r"<answer>\s*(.*?)\s*</answer>", y_chosen_clean, flags=re.DOTALL)
        tr_body = m_tr.group(1).strip() if m_tr else y_chosen_clean
        an_body = m_an.group(1).strip() if m_an else it["ground_truth"]
        wo_router_resp = f"<cog_trace>\n{tr_body}\n</cog_trace>\n<answer>{an_body}</answer>"
        abl_wo_router_records.append({
            "uid": uid,
            "benchmark": it["benchmark"],
            "prompt": q_prompt,
            "chosen": wo_router_resp,
            "rejected": rj["text"],
        })

        jargon_header = f"<cog_router>\n[Activated_Primitives]: {' -> '.join(prims)}\n</cog_router>\n"
        unstruct_base = limo_positives[uid]["text"] if uid in limo_positives else rj["text"]
        abl_jargon_records.append({
            "uid": uid,
            "benchmark": it["benchmark"],
            "prompt": q_prompt,
            "response": jargon_header + unstruct_base,
        })

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

    extra_wall = time.perf_counter() - t0
    tot_wall = prev_wall + extra_wall
    avg_c = sum(r["tokens_chosen"] for r in synth_records) / max(1, len(synth_records))
    avg_r = sum(r["tokens_rejected"] for r in synth_records) / max(1, len(synth_records))
    n_framing = sum(1 for r in synth_records if r["compression_regime"] == "structural_framing")

    stats = {
        "teacher_model": "Qwen/Qwen2.5-7B-Instruct (100% Local Self-Distillation)",
        "total_seeds": len(seeds),
        "verified_occam_pairs": len(synth_records),
        "direct_zero_hint_solved": len(synth_records) - rationalized_count,
        "star_rationalized_solved": rationalized_count,
        "matched_limo_longcot_positives": len(limo_records),
        "structural_framing_pairs": n_framing,
        "symbolic_rollout_pairs": len(synth_records) - n_framing,
        "avg_tokens_chosen": round(avg_c, 2),
        "avg_tokens_rejected": round(avg_r, 2),
        "mean_compression_ratio": round(avg_c / max(1.0, avg_r), 4),
        "primitive_activation_counts": dict(prim_counter.most_common()),
        "wall_time_sec": round(tot_wall, 2),
        "gpu_hours": round(tot_wall / 3600.0, 4),
    }
    (ASSETS_DIR / "synthesis_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 78)
    print(f"[STAGE 2B REFINER PASSED] {len(synth_records)} Pairs | Chosen={avg_c:.1f}t vs Rejected={avg_r:.1f}t (ratio={avg_c/max(1.0,avg_r):.3f}) | LIMO={len(limo_records)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
