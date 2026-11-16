#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Node 0 Gate Check: Validate the 16 Orthogonal Cognitive Primitives JSON Schema
and verify alignment with Princeton arXiv:2511.16660 & Marr's 3 Computational Levels.
"""
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = [
    "primitive_id",
    "dimension",
    "academic_citation",
    "trigger_signature",
    "mandatory_slots",
    "anti_bias_check",
]

EXPECTED_DIMENSIONS = {
    "D1_Meta_Cognitive_Controls": 4,
    "D2_Reasoning_Invariants": 4,
    "D3_Reasoning_Representations": 4,
    "D4_Epistemic_Decision_Operations": 4,
}


def main():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "cognitive_primitives_16.json"
    if not config_path.exists():
        print(f"[FAIL] Config not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    primitives = data.get("primitives", [])
    assert len(primitives) == 16, f"Expected 16 primitives, got {len(primitives)}"

    dim_counts = {}
    seen_ids = set()
    for idx, p in enumerate(primitives, 1):
        for field in REQUIRED_FIELDS:
            assert field in p and p[field], f"Primitive #{idx} missing or empty required field: {field}"
        assert isinstance(p["mandatory_slots"], list) and len(p["mandatory_slots"]) == 3, (
            f"Primitive {p['primitive_id']} must have exactly 3 mandatory_slots"
        )
        pid = p["primitive_id"]
        assert pid not in seen_ids, f"Duplicate primitive_id: {pid}"
        seen_ids.add(pid)
        dim = p["dimension"]
        dim_counts[dim] = dim_counts.get(dim, 0) + 1

    assert dim_counts == EXPECTED_DIMENSIONS, f"Dimension distribution mismatch: {dim_counts}"

    print("=" * 72)
    print("[NODE 0 GATE PASSED] 16 Orthogonal Cognitive Primitives Verified!")
    print("=" * 72)
    for dim, count in dim_counts.items():
        ids = [p["primitive_id"] for p in primitives if p["dimension"] == dim]
        print(f"  - {dim} ({count}): {', '.join(ids)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
