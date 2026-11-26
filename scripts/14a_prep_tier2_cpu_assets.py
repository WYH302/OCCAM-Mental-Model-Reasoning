#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 14A (CPU-Only, 0.00 GB GPU VRAM):
  1. Generates 5 Decoupled / Layer-Selective WiSE-FT LoRA checkpoints from OCCAM-Qwen-7B-v2-SFT
     (ZERO extra training GPU seconds!).
  2. Constructs strictly size-locked (N=1,570, ZERO training data expansion!) primitive-cardinality
     ablation datasets M in {4, 8, 16} with ultra-compact ~10-12 token <cog_router> encoding.
"""
import collections
import json
import re
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_ROOT = BASE_DIR / "outputs" / "checkpoints_occam"
SYNTH_PATH = BASE_DIR / "data" / "synthesized_pairs" / "synthesized_pairs_1600.jsonl"
SPLITS_DIR = BASE_DIR / "data" / "final_splits"
TIER2_DATA = BASE_DIR / "data" / "tier2_surgical"
TIER2_DATA.mkdir(parents=True, exist_ok=True)

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

# Exact 8 active primitives vs 4 dimension archetypes for Surgery 6 cardinality ablation
ACTIVE_8_MAP = {
    "META_1": "META_1", "META_2": "META_2", "META_3": "META_3", "META_4": "META_4",
    "INV_1": "INV_1", "INV_2": "INV_1", "INV_3": "INV_3", "INV_4": "INV_3",
    "REP_1": "REP_1", "REP_2": "REP_1", "REP_3": "REP_3", "REP_4": "REP_3",
    "OP_1": "OP_1", "OP_2": "OP_1", "OP_3": "OP_1", "OP_4": "OP_1",
}
DIM_4_MAP = {
    "META": "D1_META",
    "INV": "D2_INV",
    "REP": "D3_REP",
    "OP": "D4_OP",
}


def build_decoupled_wise_adapters():
    src = CKPT_ROOT / "OCCAM-Qwen-7B-v2-SFT"
    sd = load_file(str(src / "adapter_model.safetensors"))

    configs = [
        ("OCCAM-Qwen-7B-DecWiSE-a100-f15", 1.00, 0.15, None),
        ("OCCAM-Qwen-7B-DecWiSE-a100-f25", 1.00, 0.25, None),
        ("OCCAM-Qwen-7B-DecWiSE-a100-f35", 1.00, 0.35, None),
        ("OCCAM-Qwen-7B-DecWiSE-a90-f10", 0.90, 0.10, None),
        # Layer-Selective Decoupled WiSE-FT:
        # All Attention = 1.00; Layers 0..19 FFN (factual key-value memory bank) = 0.05;
        # Layers 20..27 FFN (upper deductive reasoning & routing) = 0.80!
        ("OCCAM-Qwen-7B-LayerWiSE-top8", 1.00, 0.05, (20, 0.80)),
    ]

    for name, g_attn, g_ffn, top_ffn in configs:
        dst = CKPT_ROOT / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*"):
            if f.is_file() and f.name not in ("adapter_model.safetensors", "training_metrics.json"):
                shutil.copy2(f, dst / f.name)

        scaled = {}
        n_attn_b, n_ffn_low_b, n_ffn_top_b = 0, 0, 0
        for k, v in sd.items():
            if "lora_B" not in k:
                scaled[k] = v.clone()
                continue
            if ".self_attn." in k:
                scaled[k] = v * g_attn
                n_attn_b += 1
            elif ".mlp." in k:
                m_lay = re.search(r"\.layers\.(\d+)\.", k)
                lay_idx = int(m_lay.group(1)) if m_lay else 0
                if top_ffn is not None and lay_idx >= top_ffn[0]:
                    scaled[k] = v * top_ffn[1]
                    n_ffn_top_b += 1
                else:
                    scaled[k] = v * g_ffn
                    n_ffn_low_b += 1
            else:
                scaled[k] = v * g_attn

        save_file(scaled, str(dst / "adapter_model.safetensors"))
        meta = {
            "checkpoint_name": name,
            "base_model": "Qwen2.5-7B-Instruct",
            "stage": "Tier2_Surgery1_Decoupled_WiSE_FT",
            "gamma_attn": g_attn,
            "gamma_ffn_low": g_ffn,
            "gamma_ffn_top": top_ffn[1] if top_ffn else g_ffn,
            "top_layer_start": top_ffn[0] if top_ffn else None,
            "extra_train_gpu_sec": 0.0,
            "scaled_tensors": {"attn_B": n_attn_b, "ffn_low_B": n_ffn_low_b, "ffn_top_B": n_ffn_top_b},
        }
        (dst / "training_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  [Decoupled WiSE Ready] {name:<32} | attn={g_attn:.2f} ffn_low={g_ffn:.2f} ffn_top={meta['gamma_ffn_top']:.2f} | tensors={meta['scaled_tensors']}")


def extract_primitive_codes(y_chosen: str):
    m = re.search(r"<cog_router>(.*?)</cog_router>", y_chosen, flags=re.S)
    if not m:
        return ["INV_1"]
    rtxt = m.group(1)
    codes = re.findall(r"\b(META|INV|REP|OP)_([1-4])\b", rtxt)
    if not codes:
        return ["INV_1"]
    seen, out = set(), []
    for dim, num in codes:
        c = f"{dim}_{num}"
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:3]


def build_cardinality_ablation_datasets():
    raw = [json.loads(ln) for ln in SYNTH_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    test_items = [
        json.loads(ln)
        for ln in (SPLITS_DIR / "test_all_8benchmarks.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    test_norm = {re.sub(r"\s+", " ", t["question"].strip().lower()) for t in test_items}

    sets = {"m4": [], "m8": [], "m16": []}
    router_lens = {"orig": [], "m4": [], "m8": [], "m16": []}

    for r in raw:
        assert re.sub(r"\s+", " ", r["question"].strip().lower()) not in test_norm, "Decontamination violation!"
        sb = str(r.get("source_benchmark", ""))
        regime = "Symbolic_Rollout" if any(sub in sb for sub in SYMBOLIC_SOURCE_SUBSTRINGS) else "Structural_Framing"
        reg_short = "Sym" if regime == "Symbolic_Rollout" else "Frame"
        y = r["y_chosen"].strip()
        m_rt = re.search(r"<cog_router>.*?</cog_router>", y, flags=re.S)
        body_after_router = y[m_rt.end():].lstrip() if m_rt else y
        orig_rt_words = len(m_rt.group(0).split()) if m_rt else 0
        router_lens["orig"].append(orig_rt_words)

        codes16 = extract_primitive_codes(y)
        # M=8 collapse
        codes8 = []
        for c in codes16:
            c8 = ACTIVE_8_MAP.get(c, "INV_1")
            if c8 not in codes8:
                codes8.append(c8)
        # M=4 collapse
        codes4 = []
        for c in codes16:
            c4 = DIM_4_MAP[c.split("_")[0]]
            if c4 not in codes4:
                codes4.append(c4)

        for key, clist in [("m4", codes4), ("m8", codes8), ("m16", codes16)]:
            compact_rt = f"<cog_router>{'->'.join(clist)}|{reg_short}</cog_router>"
            router_lens[key].append(len(compact_rt.split()))
            sets[key].append({
                "uid": r["uid"],
                "benchmark": r["benchmark"],
                "regime": regime,
                "prompt": r["question"],
                "response": f"{compact_rt}\n{body_after_router}",
            })

    for key, rows in sets.items():
        p = TIER2_DATA / f"ablation_{key}_compact.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  [Surgery 6 Dataset] {p.name:<28} | N={len(rows)} (ZERO data expansion!) | avg router words={sum(router_lens[key])/len(rows):.1f} (vs orig {sum(router_lens['orig'])/len(rows):.1f})")


if __name__ == "__main__":
    print("=== [CPU PREP 1] Building 5 Decoupled / Layer-Selective WiSE-FT Adapters (0 GPU sec) ===")
    build_decoupled_wise_adapters()
    print("\n=== [CPU PREP 2] Building Size-Locked (N=1,570) M={4,8,16} Compact-Router Datasets ===")
    build_cardinality_ablation_datasets()
    print("\n[STAGE 14A COMPLETE] Zero GPU VRAM touched!")
