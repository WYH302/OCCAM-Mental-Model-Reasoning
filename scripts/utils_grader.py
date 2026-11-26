#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Multi-Format Answer Extractor & Symbolic Equivalence Grader for OCCAM.
Eliminates formatting bias across Zero-Shot CoT, Direct 0-CoT, DeepSeek-R1 (<think> + \\boxed{}),
GSM8K (####), Multiple-Choice ((A)-(E)), FOLIO (True/False/Uncertain), and OCCAM (<answer>).
Integrates math_verify + SymPy symbolic equivalence for MATH-500 & GSM8K.
"""
import math
import re
from typing import Any, Dict, Optional, Tuple

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAS_MATH_VERIFY = True
except Exception:
    HAS_MATH_VERIFY = False

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    HAS_SYMPY = True
except Exception:
    HAS_SYMPY = False


def extract_last_boxed(text: str) -> Optional[str]:
    """Extract content of the last \\boxed{...} or \\fbox{...} with balanced brace matching."""
    idx = -1
    for cmd in ("\\boxed{", "\\fbox{"):
        pos = text.rfind(cmd)
        if pos > idx:
            idx = pos
            cmd_len = len(cmd)
    if idx == -1:
        return None
    i = idx + cmd_len
    depth = 1
    chars = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                chars.append(ch)
        else:
            chars.append(ch)
        i += 1
    return "".join(chars).strip() if depth == 0 else None


def normalize_verdict(s: str) -> str:
    s_clean = re.sub(r"[^a-z0-9]", "", s.strip().lower())
    mapping = {
        "yes": "yes",
        "true": "true",
        "no": "no",
        "false": "false",
        "valid": "valid",
        "deductivelyvalid": "valid",
        "invalid": "invalid",
        "deductivelyinvalid": "invalid",
        "uncertain": "uncertain",
        "unknown": "uncertain",
        "undetermined": "uncertain",
        "neither": "uncertain",
    }
    return mapping.get(s_clean, s_clean)


def extract_final_answer(
    raw_text: str,
    benchmark: str = "",
    ground_truth: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format-agnostic extraction of model's final answer.
    Never penalizes base/RL models for omitting <answer> XML tags.
    """
    if not raw_text or not raw_text.strip():
        return ""

    # Prefer post-</think> completion if present and non-empty
    if "</think>" in raw_text:
        after_think = raw_text.split("</think>")[-1].strip()
        work_text = after_think if len(after_think) >= 1 else raw_text
    else:
        work_text = raw_text

    # 1. Explicit <answer>...</answer> tag (last occurrence)
    ans_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", work_text, flags=re.IGNORECASE | re.DOTALL)
    if ans_matches:
        cand = ans_matches[-1].strip()
        boxed_in_cand = extract_last_boxed(cand)
        if boxed_in_cand:
            return boxed_in_cand
        return cand

    # 2. LaTeX \boxed{...}
    boxed = extract_last_boxed(work_text)
    if boxed:
        return boxed
    if work_text is not raw_text:
        boxed_full = extract_last_boxed(raw_text)
        if boxed_full:
            return boxed_full

    # 3. GSM8K #### marker
    hash_matches = re.findall(r"####\s*([^\n]+)", work_text)
    if hash_matches:
        return hash_matches[-1].strip().rstrip(".")

    # 4. Explicit conclusion phrases (check last 600 chars)
    tail = work_text[-800:]
    phrase_patterns = [
        r"(?:final answer|the answer|correct answer|correct option|conclusion|verdict)\s*(?:is|:|=)\s*\**([^\n\.\*]+)",
        r"(?:therefore|thus|hence|so),?\s+the\s+(?:final\s+)?answer\s+is\s*\**([^\n\.\*]+)",
    ]
    for pat in phrase_patterns:
        m_list = re.findall(pat, tail, flags=re.IGNORECASE)
        if m_list:
            cand = m_list[-1].strip().strip("*:# ").strip()
            if cand and re.search(r"[a-zA-Z0-9]", cand):
                return cand

    gt_clean = ground_truth.strip()

    # 5. Task-type aware fallback on the concluding lines
    last_lines = "\n".join([ln.strip() for ln in tail.splitlines() if ln.strip()][-4:])

    # 5a. Multiple Choice (A/B/C/D/E or (A)-(E))
    if re.fullmatch(r"\(?[A-Ea-e]\)?", gt_clean):
        mc_pats = [
            r"\(([A-Ea-e])\)",
            r"\bOption\s+([A-Ea-e])\b",
            r"\b([A-E])\b",
        ]
        for mp in mc_pats:
            ms = re.findall(mp, last_lines)
            if ms:
                return ms[-1].upper()
        # Check if exact option text was written out
        if metadata and isinstance(metadata.get("choices"), dict):
            for k, opt_txt in metadata["choices"].items():
                if opt_txt and len(str(opt_txt)) > 2 and str(opt_txt).lower() in last_lines.lower():
                    return str(k).strip("()").upper()

    # 5b. Yes/No, Valid/Invalid, True/False/Uncertain
    gt_verdict = normalize_verdict(gt_clean)
    if gt_verdict in ("yes", "no", "valid", "invalid", "true", "false", "uncertain"):
        v_matches = re.findall(
            r"\b(yes|no|deductively valid|deductively invalid|valid|invalid|true|false|uncertain)\b",
            last_lines,
            flags=re.IGNORECASE,
        )
        if v_matches:
            return v_matches[-1]

    # 5c. Numeric / Math fallback: last number or fraction in concluding lines
    if benchmark in ("MATH-500", "GSM8K", "AIME-2024", "AMC-2023") or re.search(r"\d", gt_clean):
        num_matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", last_lines)
        if num_matches:
            return num_matches[-1]

    # Default: last non-empty line
    lines = [ln.strip() for ln in work_text.splitlines() if ln.strip()]
    return lines[-1][:120] if lines else ""


def clean_math_string(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("$", "").replace(" ", "")
    # Remove thousands commas between digits
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return s.rstrip(".")


def try_float(s: str) -> Optional[float]:
    s_c = clean_math_string(s)
    s_c = re.sub(r"[a-zA-Z%]+$", "", s_c).strip()
    try:
        if "/" in s_c and not any(c in s_c for c in ("\\", "{", "}")):
            parts = s_c.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        return float(s_c)
    except Exception:
        return None


def latex_to_sympy_friendly(s: str) -> str:
    s = clean_math_string(s)
    # \frac{a}{b} -> ((a)/(b))
    for _ in range(5):
        s_new = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
        if s_new == s:
            break
        s = s_new
    for _ in range(4):
        s_new = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
        if s_new == s:
            break
        s = s_new
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
    return s


def grade_math_equivalence(pred: str, gt: str) -> bool:
    if not pred:
        return False
    p_clean = clean_math_string(pred)
    g_clean = clean_math_string(gt)
    if p_clean.lower() == g_clean.lower():
        return True

    # Numeric comparison
    f_p = try_float(p_clean)
    f_g = try_float(g_clean)
    if f_p is not None and f_g is not None:
        if math.isclose(f_p, f_g, rel_tol=1e-4, abs_tol=1e-4):
            return True

    # math_verify check
    if HAS_MATH_VERIFY:
        try:
            parsed_gt = mv_parse(f"${gt}$")
            parsed_pred = mv_parse(f"${pred}$")
            if parsed_gt and parsed_pred and mv_verify(parsed_gt, parsed_pred):
                return True
        except Exception:
            pass

    # SymPy symbolic simplification check
    if HAS_SYMPY:
        try:
            tr = standard_transformations + (implicit_multiplication_application, convert_xor)
            e_p = parse_expr(latex_to_sympy_friendly(pred), transformations=tr, evaluate=True)
            e_g = parse_expr(latex_to_sympy_friendly(gt), transformations=tr, evaluate=True)
            diff = sp.simplify(e_p - e_g)
            if diff == 0:
                return True
            if e_p.is_number and e_g.is_number:
                if abs(complex(e_p.evalf()) - complex(e_g.evalf())) < 1e-4:
                    return True
        except Exception:
            pass

    return False


def grade_prediction(
    raw_output: str,
    ground_truth: str,
    benchmark: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    Unified grader returning (is_correct, extracted_answer).
    """
    extracted = extract_final_answer(raw_output, benchmark=benchmark, ground_truth=ground_truth, metadata=metadata)
    if not extracted:
        return False, ""
    extracted = re.sub(r"\\(?:text|mathrm|textbf|mbox|mathbf)\{([^}]*)\}", r"\1", extracted).strip()

    gt = ground_truth.strip()

    # 1. Multiple Choice Letters (e.g., A, (B), etc.)
    gt_mc = re.fullmatch(r"\(?([A-Ea-e])\)?", gt)
    if gt_mc:
        gt_letter = gt_mc.group(1).upper()
        # Extract leading letter from prediction like "(B) ..." or "B." or "Option B"
        m_pred = re.match(r"^\s*(?:option\s+)?\(?([A-Ea-e])\)?(?:[\.\)\s:]|$)", extracted.strip(), flags=re.IGNORECASE)
        if m_pred:
            return (m_pred.group(1).upper() == gt_letter), m_pred.group(1).upper()
        if extracted.strip().upper() == gt_letter:
            return True, gt_letter
        # Also check exact option text match if choices provided
        if metadata and isinstance(metadata.get("choices"), dict):
            opt_text = str(metadata["choices"].get(gt_letter, "")).strip().lower()
            if opt_text and opt_text == extracted.strip().lower().rstrip("."):
                return True, gt_letter
        return False, extracted[:40]

    # 2. Categorical Logic Verdicts (Yes/No, Valid/Invalid, True/False/Uncertain)
    v_gt = normalize_verdict(gt)
    if v_gt in ("yes", "no", "valid", "invalid", "true", "false", "uncertain"):
        v_pred = normalize_verdict(extracted)
        if v_pred == v_gt:
            return True, extracted[:40]
        # Check first token of extracted
        m_v = re.search(r"\b(yes|no|valid|invalid|true|false|uncertain)\b", extracted, flags=re.IGNORECASE)
        if m_v and normalize_verdict(m_v.group(1)) == v_gt:
            return True, m_v.group(1)
        return False, extracted[:40]

    # 3. Mathematical / Symbolic / Numeric Equivalence
    if grade_math_equivalence(extracted, gt):
        return True, extracted[:80]

    # 4. Normalized exact text fallback
    norm_p = re.sub(r"\s+", " ", extracted.strip().lower().rstrip("."))
    norm_g = re.sub(r"\s+", " ", gt.lower().rstrip("."))
    return (norm_p == norm_g), extracted[:80]
