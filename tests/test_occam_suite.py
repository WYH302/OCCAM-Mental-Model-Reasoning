import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_cognitive_primitives_schema():
    cfg_path = ROOT / "configs" / "cognitive_primitives_16.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    prims = cfg.get("primitives", [])
    assert len(prims) == 16
    ids = {p["primitive_id"] for p in prims}
    assert len(ids) == 16
    for p in prims:
        assert len(p.get("mandatory_slots", [])) == 3


def test_retained_metrics_integrity():
    eval_path = ROOT / "results" / "eval_summary_metrics.json"
    synth_path = ROOT / "results" / "synthesis_stats.json"
    assert eval_path.exists() and synth_path.exists()
    ev = json.loads(eval_path.read_text(encoding="utf-8"))
    sy = json.loads(synth_path.read_text(encoding="utf-8"))
    assert sy["verified_occam_pairs"] == 1570
    assert "OCCAM-7B-SFT" in ev["models"]
    assert "R1-Distill-7B" in ev["models"]


def test_dataset_parquet_files_present():
    ds_dir = ROOT / "dataset" / "OCCAM-2K"
    for split_fn in [
        "train-00000-of-00001.parquet",
        "test_id-00000-of-00001.parquet",
        "test_ood-00000-of-00001.parquet",
    ]:
        p = ds_dir / split_fn
        assert p.exists() and p.stat().st_size > 10000
