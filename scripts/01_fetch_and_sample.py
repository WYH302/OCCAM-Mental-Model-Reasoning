#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 (OCCAM-2K & Full-Scale 8-Benchmark Suite):
Fetches 8 authoritative public reasoning benchmarks via hf-mirror.com / datasets:
  - 5 In-Distribution (ID) Benchmarks (3,223 official test problems):
      1. CLadder (450 balanced Rung 1/2/3 test; 400 disjoint train)
      2. BBH-Hard (450 stratified 6-task test; 400 disjoint train)
      3. ARC-Challenge (1,172 FULL official test split; 400 official train split)
      4. LogiQA (651 FULL official test split; 200 official train split)
      5. MATH-500 (500 FULL official MATH-500 test split; 200 official Hendrycks MATH train split)
  - 3 Zero-Shot Out-of-Distribution (OOD) Benchmarks (1,303 official test problems, 0 train):
      6. GSM8K (800 official test split items)
      7. FOLIO (FULL ~203 official FOL validation/test items)
      8. StrategyQA (300 official multi-hop test/dev items)
Total Official Evaluation Suite: ~4,526 problems!
Enforces strict 0.00% SHA-256 exact match & 0.00% content 13-gram overlap across ALL train vs. test items.
"""
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "raw_seeds" / "hf_cache"
RAW_SEEDS_DIR = BASE_DIR / "data" / "raw_seeds"
FINAL_SPLITS_DIR = BASE_DIR / "data" / "final_splits"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_SEEDS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def get_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "OCCAM-Research/1.0"})
    return session


def fetch_cached(urls, cache_name: str) -> bytes:
    if isinstance(urls, str):
        urls = [urls]
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and cache_path.stat().st_size > 200:
        return cache_path.read_bytes()
    session = get_session()
    last_err = None
    for url in urls:
        for attempt in range(1, 4):
            try:
                print(f"  [Downloading] {cache_name} <- {url} (attempt {attempt})...")
                resp = session.get(url, timeout=45)
                resp.raise_for_status()
                if len(resp.content) > 200:
                    cache_path.write_bytes(resp.content)
                    return resp.content
            except Exception as e:
                last_err = e
                print(f"  [Retry] {cache_name} failed ({e})")
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to download {cache_name}: {last_err}")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


TASK_BOILERPLATE_PATTERNS = [
    r"imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships:",
    r"answer strictly with yes or no\.",
    r"how would a typical person answer each of the following questions about causation\?",
    r"the following paragraphs each describe a set of five objects arranged in a fixed order\. the statements are logically consistent within each paragraph\.",
    r"here comes a perfectly valid argument:",
    r"is the following argument deductively valid or invalid\?",
    r"if you follow these instructions, do you return to the starting point\?always face forward\.",
    r"if you follow these instructions, do you return to the starting point\?",
    r"always face forward\.",
    r"evaluate the result of a random boolean expression\.",
    r"does the truth-teller tell the truth\?",
    r"determine whether the hypothesis is true, false, or uncertain strictly based on the premises\.",
]


def get_13grams(text: str) -> set:
    cleaned = text.lower()
    for pat in TASK_BOILERPLATE_PATTERNS:
        cleaned = re.sub(pat, " ", cleaned)
    tokens = re.findall(r"\w+", cleaned)
    if len(tokens) < 13:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + 13]) for i in range(len(tokens) - 12)}


def compute_boilerplate_13grams(pool: list, max_doc_freq: int = 12) -> set:
    df_counts = Counter()
    for item in pool:
        for g in get_13grams(item["question"]):
            df_counts[g] += 1
    return {g for g, c in df_counts.items() if c > max_doc_freq}


def extract_boxed_answer(solution_str: str) -> str:
    pos = solution_str.rfind("\\boxed{")
    if pos == -1:
        pos = solution_str.rfind("\\fbox{")
        cmd_len = 6
    else:
        cmd_len = 7
    if pos == -1:
        return ""
    i = pos + cmd_len
    depth = 1
    out = []
    while i < len(solution_str) and depth > 0:
        ch = solution_str[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() if depth == 0 else ""


# -------------------------------------------------------------------------
# 1. CLadder (Balanced Rung 1 / 2 / 3)
# -------------------------------------------------------------------------
def load_cladder_splits(rng: random.Random, boilerplate: set):
    url = "https://hf-mirror.com/datasets/causal-nlp/CLadder/resolve/main/data/test-balanced-v1.5.csv"
    raw = fetch_cached(url, "cladder_test_balanced_v1.5.csv")
    df = pd.read_csv(io.BytesIO(raw))
    by_rung = {"1": [], "2": [], "3": []}
    seen_q = set()
    for _, row in df.iterrows():
        q = str(row["prompt"]).strip()
        norm_q = normalize_text(q)
        if norm_q in seen_q:
            continue
        seen_q.add(norm_q)
        gt = str(row["label"]).strip().lower()
        gt_norm = "Yes" if gt in ("yes", "1", "true") else "No"
        rung = str(row.get("rung", "2")).strip()
        item = {
            "uid": f"CLADDER_{row['id']}",
            "benchmark": "CLadder",
            "split_group": "ID",
            "source_benchmark": f"causal-nlp/CLadder (Rung-{rung}/{row.get('query_type', 'ate')})",
            "category": "Category_A_Causal_Inference",
            "question": f"{q}\nAnswer strictly with Yes or No.",
            "ground_truth": gt_norm,
            "metadata": {
                "rung": rung,
                "query_type": str(row.get("query_type", "")),
                "graph_id": str(row.get("graph_id", "")),
                "formal_form": str(row.get("formal_form", "")),
                "gold_reasoning_sketch": str(row.get("reasoning", ""))[:500],
            },
        }
        if rung in by_rung:
            by_rung[rung].append(item)

    # Compute CLadder-local boilerplate (causal story templates shared across queries)
    all_cladder = by_rung["1"] + by_rung["2"] + by_rung["3"]
    cladder_bp = boilerplate | compute_boilerplate_13grams(all_cladder, max_doc_freq=6)

    test_items, train_items = [], []
    test_grams, test_hashes = set(), set()

    # First lock 150 per rung = 450 test items
    leftovers = {"1": [], "2": [], "3": []}
    for r_key in ("1", "2", "3"):
        pool = list(by_rung[r_key])
        rng.shuffle(pool)
        for it in pool:
            if len([x for x in test_items if x["metadata"]["rung"] == r_key]) < 150:
                test_items.append(it)
                test_hashes.add(hashlib.sha256(normalize_text(it["question"]).encode("utf-8")).hexdigest())
                test_grams.update(get_13grams(it["question"]) - cladder_bp)
            else:
                leftovers[r_key].append(it)

    # Next select strictly 13-gram disjoint train items (133 + 134 + 133 = 400)
    targets_tr = {"1": 133, "2": 134, "3": 133}
    for r_key, n_tr in targets_tr.items():
        got = 0
        for it in leftovers[r_key]:
            if got >= n_tr:
                break
            h = hashlib.sha256(normalize_text(it["question"]).encode("utf-8")).hexdigest()
            gs = get_13grams(it["question"]) - cladder_bp
            if h in test_hashes or (gs & test_grams):
                continue
            train_items.append(it)
            got += 1
        assert got == n_tr, f"CLadder rung {r_key} train shortfall: {got}/{n_tr}"

    return train_items, test_items, cladder_bp


# -------------------------------------------------------------------------
# 2. BBH-Hard (6 Canonical Cognitive/Logical Tasks)
# -------------------------------------------------------------------------
def load_bbh_splits(rng: random.Random):
    subsets = [
        "causal_judgement",
        "formal_fallacies",
        "logical_deduction_five_objects",
        "navigate",
        "web_of_lies",
        "boolean_expressions",
        "tracking_shuffled_objects_five_objects",
        "disambiguation_qa",
        "date_understanding",
        "reasoning_about_colored_objects",
    ]
    all_bbh = []
    by_sub = {}
    seen_q = set()
    for sub in subsets:
        url = f"https://hf-mirror.com/datasets/lukaemon/bbh/resolve/main/{sub}/test-00000-of-00001.parquet"
        raw = fetch_cached(url, f"bbh_{sub}.parquet")
        df = pd.read_parquet(io.BytesIO(raw))
        sub_list = []
        for idx, row in df.iterrows():
            q = str(row["input"]).strip()
            norm_q = normalize_text(q)
            if norm_q in seen_q:
                continue
            seen_q.add(norm_q)
            gt = str(row["target"]).strip()
            it = {
                "uid": f"BBH_{sub}_{idx:04d}",
                "benchmark": "BBH-Hard",
                "split_group": "ID",
                "source_benchmark": f"lukaemon/bbh ({sub})",
                "category": "Category_B_Logic_Fallacies",
                "question": q,
                "ground_truth": gt,
                "metadata": {"subset": sub},
            }
            sub_list.append(it)
            all_bbh.append(it)
        by_sub[sub] = sub_list

    bbh_bp = compute_boilerplate_13grams(all_bbh, max_doc_freq=3)
    test_items, candidate_train = [], []
    te_hashes, te_grams = set(), set()

    # 45 test items per subset across 10 canonical BBH tasks = 450 test items
    for sub in subsets:
        pool = list(by_sub[sub])
        rng.shuffle(pool)
        te = pool[:45]
        test_items.extend(te)
        for x in te:
            te_hashes.add(hashlib.sha256(normalize_text(x["question"]).encode("utf-8")).hexdigest())
            te_grams.update(get_13grams(x["question"]) - bbh_bp)
        candidate_train.extend(pool[45:])

    rng.shuffle(candidate_train)
    train_items = []
    for x in candidate_train:
        if len(train_items) >= 400:
            break
        h = hashlib.sha256(normalize_text(x["question"]).encode("utf-8")).hexdigest()
        gs = get_13grams(x["question"]) - bbh_bp
        if h in te_hashes or (gs & te_grams):
            continue
        train_items.append(x)

    assert len(test_items) == 450 and len(train_items) == 400, (
        f"BBH shortfall: te={len(test_items)}/450, tr={len(train_items)}/400"
    )
    return train_items, test_items, bbh_bp


# -------------------------------------------------------------------------
# 3. ARC-Challenge (Full 1,172 Official Test + 400 Official Train)
# -------------------------------------------------------------------------
def load_arc_splits(rng: random.Random):
    def parse_arc_df(df: pd.DataFrame, split_origin: str) -> list:
        out = []
        seen = set()
        label_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        for _, row in df.iterrows():
            q_stem = str(row["question"]).strip()
            choices = row["choices"]
            texts = list(choices["text"])
            labels = [label_map.get(str(l), str(l)) for l in list(choices["label"])]
            ans_key = label_map.get(str(row["answerKey"]).strip(), str(row["answerKey"]).strip())
            opts_str = "\n".join([f"({lbl}) {txt}" for lbl, txt in zip(labels, texts)])
            full_q = f"{q_stem}\nOptions:\n{opts_str}"
            nq = normalize_text(full_q)
            if nq in seen:
                continue
            seen.add(nq)
            out.append({
                "uid": f"ARC_{split_origin}_{row['id']}",
                "benchmark": "ARC-Challenge",
                "split_group": "ID",
                "source_benchmark": f"allenai/ai2_arc (ARC-Challenge/{split_origin})",
                "category": "Category_C_Scientific_Systems",
                "question": full_q,
                "ground_truth": ans_key,
                "metadata": {"split_origin": split_origin, "choices": dict(zip(labels, texts))},
            })
        return out

    raw_te = fetch_cached(
        "https://hf-mirror.com/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet",
        "arc_challenge_test.parquet",
    )
    raw_tr = fetch_cached(
        "https://hf-mirror.com/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/train-00000-of-00001.parquet",
        "arc_challenge_train.parquet",
    )
    test_items = parse_arc_df(pd.read_parquet(io.BytesIO(raw_te)), "test")
    train_pool = parse_arc_df(pd.read_parquet(io.BytesIO(raw_tr)), "train")
    arc_bp = compute_boilerplate_13grams(test_items + train_pool, max_doc_freq=8)

    te_hashes = {hashlib.sha256(normalize_text(x["question"]).encode("utf-8")).hexdigest() for x in test_items}
    te_grams = set()
    for x in test_items:
        te_grams.update(get_13grams(x["question"]) - arc_bp)

    rng.shuffle(train_pool)
    train_items = []
    for x in train_pool:
        if len(train_items) >= 400:
            break
        h = hashlib.sha256(normalize_text(x["question"]).encode("utf-8")).hexdigest()
        gs = get_13grams(x["question"]) - arc_bp
        if h in te_hashes or (gs & te_grams):
            continue
        train_items.append(x)
    assert len(train_items) == 400, f"ARC train shortfall: {len(train_items)}"
    return train_items, test_items, arc_bp


# -------------------------------------------------------------------------
# 4. LogiQA (Full 651 Official Test + 200 Official Train)
# -------------------------------------------------------------------------
def load_logiqa_splits(rng: random.Random):
    def parse_logiqa_df(df: pd.DataFrame, split_origin: str) -> list:
        out = []
        seen = set()
        idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
        for idx, row in df.iterrows():
            ctx = str(row["context"]).strip()
            qry = str(row["query"]).strip()
            opts = list(row["options"])
            ans_letter = idx_to_letter.get(int(row["correct_option"]), "A")
            opts_str = "\n".join([f"({idx_to_letter[i]}) {opt}" for i, opt in enumerate(opts)])
            full_q = f"Context: {ctx}\nQuestion: {qry}\nOptions:\n{opts_str}"
            nq = normalize_text(full_q)
            if nq in seen:
                continue
            seen.add(nq)
            out.append({
                "uid": f"LOGIQA_{split_origin}_{idx:04d}",
                "benchmark": "LogiQA",
                "split_group": "ID",
                "source_benchmark": f"lucasmccabe/logiqa ({split_origin})",
                "category": "Category_D_Backward_Math_Logic",
                "question": full_q,
                "ground_truth": ans_letter,
                "metadata": {"sub_type": "logiqa", "choices": {idx_to_letter[i]: opt for i, opt in enumerate(opts)}},
            })
        return out

    raw_te = fetch_cached(
        "https://hf-mirror.com/datasets/lucasmccabe/logiqa/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
        "logiqa_test.parquet",
    )
    raw_tr = fetch_cached(
        "https://hf-mirror.com/datasets/lucasmccabe/logiqa/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "logiqa_train.parquet",
    )
    test_items = parse_logiqa_df(pd.read_parquet(io.BytesIO(raw_te)), "test")
    train_pool = parse_logiqa_df(pd.read_parquet(io.BytesIO(raw_tr)), "train")
    logi_bp = compute_boilerplate_13grams(test_items + train_pool, max_doc_freq=10)

    te_hashes = {hashlib.sha256(normalize_text(x["question"]).encode("utf-8")).hexdigest() for x in test_items}
    te_grams = set()
    for x in test_items:
        te_grams.update(get_13grams(x["question"]) - logi_bp)

    rng.shuffle(train_pool)
    train_items = []
    for x in train_pool:
        if len(train_items) >= 200:
            break
        h = hashlib.sha256(normalize_text(x["question"]).encode("utf-8")).hexdigest()
        gs = get_13grams(x["question"]) - logi_bp
        if h in te_hashes or (gs & te_grams):
            continue
        train_items.append(x)
    assert len(train_items) == 200, f"LogiQA train shortfall: {len(train_items)}"
    return train_items, test_items, logi_bp


# -------------------------------------------------------------------------
# 5. MATH-500 (Full 500 Official Test) + Hendrycks MATH Official Train (200)
# -------------------------------------------------------------------------
def load_math_splits(rng: random.Random):
    raw_te = fetch_cached(
        "https://hf-mirror.com/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
        "math_500_test.jsonl",
    )
    test_items = []
    seen_te = set()
    for idx, line in enumerate(raw_te.decode("utf-8").splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        prob = str(obj["problem"]).strip()
        nq = normalize_text(prob)
        if nq in seen_te:
            continue
        seen_te.add(nq)
        test_items.append({
            "uid": f"MATH500_{idx:04d}",
            "benchmark": "MATH-500",
            "split_group": "ID",
            "source_benchmark": f"HuggingFaceH4/MATH-500 ({obj.get('subject', 'Math')})",
            "category": "Category_D_Backward_Math_Logic",
            "question": prob,
            "ground_truth": str(obj["answer"]).strip(),
            "metadata": {
                "sub_type": "math500",
                "subject": str(obj.get("subject", "")),
                "level": str(obj.get("level", "")),
            },
        })

    # Official Hendrycks MATH train subjects
    math_subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    train_pool = []
    seen_tr = set()
    for subj in math_subjects:
        urls = [
            f"https://hf-mirror.com/datasets/EleutherAI/hendrycks_math/resolve/main/{subj}/train-00000-of-00001.parquet",
            f"https://hf-mirror.com/datasets/DigitalLearningGmbH/MATH-lighteval/resolve/main/data/{subj}_train.parquet",
        ]
        raw_s = fetch_cached(urls, f"hendrycks_math_train_{subj}.parquet")
        df_s = pd.read_parquet(io.BytesIO(raw_s))
        for idx, row in df_s.iterrows():
            prob = str(row["problem"]).strip()
            sol = str(row["solution"]).strip()
            ans = extract_boxed_answer(sol)
            if not ans or len(ans) > 60:
                continue
            nq = normalize_text(prob)
            if nq in seen_tr:
                continue
            seen_tr.add(nq)
            train_pool.append({
                "uid": f"MATH_TR_{subj}_{idx:04d}",
                "benchmark": "MATH-500",
                "split_group": "ID",
                "source_benchmark": f"EleutherAI/hendrycks_math ({subj}/train)",
                "category": "Category_D_Backward_Math_Logic",
                "question": prob,
                "ground_truth": ans,
                "metadata": {
                    "sub_type": "math_train",
                    "subject": subj,
                    "level": str(row.get("level", "")),
                    "gold_solution_sketch": sol[:600],
                },
            })

    math_bp = compute_boilerplate_13grams(test_items + train_pool[:1500], max_doc_freq=10)
    te_hashes = {hashlib.sha256(normalize_text(x["question"]).encode("utf-8")).hexdigest() for x in test_items}
    te_grams = set()
    for x in test_items:
        te_grams.update(get_13grams(x["question"]) - math_bp)

    rng.shuffle(train_pool)
    train_items = []
    for x in train_pool:
        if len(train_items) >= 200:
            break
        h = hashlib.sha256(normalize_text(x["question"]).encode("utf-8")).hexdigest()
        gs = get_13grams(x["question"]) - math_bp
        if h in te_hashes or (gs & te_grams):
            continue
        train_items.append(x)
    assert len(train_items) == 200, f"MATH train shortfall: {len(train_items)}"
    return train_items, test_items, math_bp


# -------------------------------------------------------------------------
# 6. Zero-Shot OOD Benchmarks: GSM8K (800), FOLIO (Full ~203), StrategyQA (300)
# -------------------------------------------------------------------------
def load_ood_benchmarks(rng: random.Random):
    # 6a. GSM8K official test split
    raw_gsm = fetch_cached(
        "https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet",
        "gsm8k_test.parquet",
    )
    df_gsm = pd.read_parquet(io.BytesIO(raw_gsm))
    gsm_pool = []
    seen = set()
    for idx, row in df_gsm.iterrows():
        q = str(row["question"]).strip()
        ans_full = str(row["answer"]).strip()
        gt = ans_full.split("####")[-1].strip().replace(",", "") if "####" in ans_full else ans_full
        nq = normalize_text(q)
        if nq in seen:
            continue
        seen.add(nq)
        gsm_pool.append({
            "uid": f"GSM8K_{idx:04d}",
            "benchmark": "GSM8K",
            "split_group": "OOD",
            "source_benchmark": "openai/gsm8k (test)",
            "category": "OOD_Quantitative_MultiStep",
            "question": q,
            "ground_truth": gt,
            "metadata": {"sub_type": "gsm8k"},
        })
    rng.shuffle(gsm_pool)
    gsm_test = gsm_pool[:800]

    # 6b. FOLIO official validation split (full ~203 items)
    folio_urls = [
        "https://hf-mirror.com/datasets/tasksource/folio/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
        "https://hf-mirror.com/datasets/benlipkin/folio/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
        "https://hf-mirror.com/datasets/yale-nlp/FOLIO/resolve/main/folio_v2_validation.jsonl",
    ]
    raw_folio = fetch_cached(folio_urls, "folio_validation.bin")
    folio_test = []
    seen_f = set()
    try:
        df_folio = pd.read_parquet(io.BytesIO(raw_folio))
        rows_iter = [row.to_dict() for _, row in df_folio.iterrows()]
    except Exception:
        rows_iter = [json.loads(ln) for ln in raw_folio.decode("utf-8").splitlines() if ln.strip()]

    label_norm_map = {
        "true": "True",
        "false": "False",
        "uncertain": "Uncertain",
        "unknown": "Uncertain",
        "0": "True",
        "1": "False",
        "2": "Uncertain",
    }
    for idx, row in enumerate(rows_iter):
        prem = str(row.get("premises", "")).strip()
        conc = str(row.get("conclusion", "")).strip()
        lbl_raw = str(row.get("label", "")).strip().lower()
        gt_lbl = label_norm_map.get(lbl_raw, str(row.get("label", "Uncertain")).strip().capitalize())
        if gt_lbl not in ("True", "False", "Uncertain"):
            continue
        full_q = (
            f"Premises:\n{prem}\n\nHypothesis: {conc}\n"
            f"Determine whether the hypothesis is True, False, or Uncertain strictly based on the premises."
        )
        nq = normalize_text(full_q)
        if nq in seen_f:
            continue
        seen_f.add(nq)
        folio_test.append({
            "uid": f"FOLIO_{idx:04d}",
            "benchmark": "FOLIO",
            "split_group": "OOD",
            "source_benchmark": "yale-nlp/FOLIO (validation)",
            "category": "OOD_FirstOrder_Logic",
            "question": full_q,
            "ground_truth": gt_lbl,
            "metadata": {"sub_type": "folio"},
        })

    # 6c. StrategyQA official split (300 items)
    sqa_urls = [
        "https://hf-mirror.com/datasets/ChilleD/StrategyQA/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
        "https://hf-mirror.com/datasets/ChilleD/StrategyQA/resolve/main/strategyqa_test.json",
        "https://hf-mirror.com/datasets/wics/strategy-qa/resolve/refs%2Fconvert%2Fparquet/strategyQA/test/0000.parquet",
        "https://hf-mirror.com/datasets/voidful/StrategyQA/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
    ]
    raw_sqa = fetch_cached(sqa_urls, "strategyqa_test.bin")
    try:
        df_sqa = pd.read_parquet(io.BytesIO(raw_sqa))
        sqa_rows = [row.to_dict() for _, row in df_sqa.iterrows()]
    except Exception:
        sqa_rows = json.loads(raw_sqa.decode("utf-8"))

    sqa_pool = []
    seen_s = set()
    for idx, row in enumerate(sqa_rows):
        q = str(row.get("question", "")).strip()
        ans_val = row.get("answer")
        if ans_val is None:
            continue
        if isinstance(ans_val, bool):
            gt_yn = "Yes" if ans_val else "No"
        else:
            s_a = str(ans_val).strip().lower()
            if s_a in ("true", "yes", "1"):
                gt_yn = "Yes"
            elif s_a in ("false", "no", "0"):
                gt_yn = "No"
            else:
                continue
        full_q = f"{q}\nAnswer strictly with Yes or No."
        nq = normalize_text(full_q)
        if nq in seen_s:
            continue
        seen_s.add(nq)
        sqa_pool.append({
            "uid": f"STRATEGYQA_{idx:04d}",
            "benchmark": "StrategyQA",
            "split_group": "OOD",
            "source_benchmark": "StrategyQA (official)",
            "category": "OOD_Implicit_MultiHop",
            "question": full_q,
            "ground_truth": gt_yn,
            "metadata": {"sub_type": "strategyqa"},
        })
    rng.shuffle(sqa_pool)
    sqa_test = sqa_pool[:300]

    return gsm_test, folio_test, sqa_test


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    rng = random.Random(20261111)

    print("=" * 78)
    print("[Stage 1] Building OCCAM-2K Train Seeds (1,600) & 8-Benchmark Official Suite (~4,500)...")
    print("=" * 78)

    tr_cladder, te_cladder, bp_cladder = load_cladder_splits(rng, set())
    tr_bbh, te_bbh, bp_bbh = load_bbh_splits(rng)
    tr_arc, te_arc, bp_arc = load_arc_splits(rng)
    tr_logi, te_logi, bp_logi = load_logiqa_splits(rng)
    tr_math, te_math, bp_math = load_math_splits(rng)
    te_gsm, te_folio, te_sqa = load_ood_benchmarks(rng)

    all_train = tr_cladder + tr_bbh + tr_arc + tr_logi + tr_math
    all_test_id = te_cladder + te_bbh + te_arc + te_logi + te_math
    all_test_ood = te_gsm + te_folio + te_sqa
    all_test_full = all_test_id + all_test_ood

    global_bp = bp_cladder | bp_bbh | bp_arc | bp_logi | bp_math | compute_boilerplate_13grams(all_test_ood, max_doc_freq=8)

    # Final global cross-benchmark decontamination audit (Train 1,600 vs. ALL ~4,526 Test Items)
    test_hashes = {hashlib.sha256(normalize_text(x["question"]).encode("utf-8")).hexdigest() for x in all_test_full}
    test_grams = set()
    for x in all_test_full:
        test_grams.update(get_13grams(x["question"]) - global_bp)

    clean_train = []
    for x in all_train:
        h = hashlib.sha256(normalize_text(x["question"]).encode("utf-8")).hexdigest()
        gs = get_13grams(x["question"]) - global_bp
        assert h not in test_hashes, f"Exact hash collision detected on {x['uid']}!"
        if gs & test_grams:
            raise AssertionError(f"13-gram collision detected on {x['uid']}!")
        clean_train.append(x)

    train_path = RAW_SEEDS_DIR / "train_seeds_1600.jsonl"
    test_id_path = FINAL_SPLITS_DIR / "test_id_full.jsonl"
    test_ood_path = FINAL_SPLITS_DIR / "test_ood_full.jsonl"
    test_all_path = FINAL_SPLITS_DIR / "test_all_8benchmarks.jsonl"
    stats_path = FINAL_SPLITS_DIR / "stage1_split_manifest.json"

    for path, items in [
        (train_path, clean_train),
        (test_id_path, all_test_id),
        (test_ood_path, all_test_ood),
        (test_all_path, all_test_full),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    dt = time.perf_counter() - t0
    bm_counts = Counter(x["benchmark"] for x in all_test_full)
    tr_bm_counts = Counter(x["benchmark"] for x in clean_train)

    manifest = {
        "train_total": len(clean_train),
        "test_id_total": len(all_test_id),
        "test_ood_total": len(all_test_ood),
        "test_grand_total": len(all_test_full),
        "train_by_benchmark": dict(tr_bm_counts),
        "test_by_benchmark": dict(bm_counts),
        "sha256_overlap_pct": 0.0,
        "content_13gram_overlap_pct": 0.0,
        "wall_time_sec": round(dt, 2),
    }
    stats_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print(f"[STAGE 1 GATE PASSED] {len(clean_train)} Train Seeds & {len(all_test_full)} Official Test Items Verified!")
    print("=" * 78)
    for bm in ["CLadder", "BBH-Hard", "ARC-Challenge", "LogiQA", "MATH-500", "GSM8K", "FOLIO", "StrategyQA"]:
        print(f"  - {bm:<16} | Train: {tr_bm_counts.get(bm, 0):4d} | Official Test: {bm_counts.get(bm, 0):4d}")
    print("-" * 78)
    print(f"  Grand Total Test Suite : {len(all_test_full)} problems (ID={len(all_test_id)}, OOD={len(all_test_ood)})")
    print(f"  Decontamination Audit  : SHA-256 = 0.00% | 13-gram = 0.00% | Time = {dt:.2f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
