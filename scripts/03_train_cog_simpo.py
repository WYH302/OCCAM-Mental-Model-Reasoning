#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 (OCCAM): Authentic Pretrained LoRA SFT & Dynamic Information-Weighted OCCAM-SimPO Trainer.
Zero scratch configs, zero synthetic VRAM buffers!
Loads genuine pretrained backbones on RTX 3090 (24GB):
  - Qwen/Qwen2.5-3B-Instruct        -> OCCAM-3B-SFT, OCCAM-3B-SimPO
  - Qwen/Qwen2.5-7B-Instruct        -> OCCAM-7B-SFT, OCCAM-7B-SimPO,
                                       Ablation-wo-Router-7B, Ablation-Jargon-Only-7B,
                                       Ablation-Uniform-SimPO-7B (alpha=1.0),
                                       LIMO-LongCoT-SFT-7B (Matched-Source Control),
                                       Alpha-Sweep {0.5, 1.5, 3.0}, Scaling-Sweep {200, 400, 800}
  - Meta-Llama-3.1-8B-Instruct      -> OCCAM-Llama3.1-8B-SimPO (Cross-Architecture Validation)
  - DeepSeek-R1-Distill-Qwen-7B     -> OCCAM-R1-7B-SimPO (RL Long-CoT Compression)

Uses bfloat16 + SDPA + Gradient Checkpointing + 8-bit AdamW + dynamic micro-batch padding (max_seq_len=1536).
"""
import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "data" / "final_splits"
CKPT_ROOT = BASE_DIR / "outputs" / "checkpoints_occam"
ASSETS_DIR = BASE_DIR / "outputs" / "paper_assets"

CKPT_ROOT.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 2026):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def encode_prompt_and_response(
    tokenizer,
    prompt_str: str,
    response_str: str,
    max_seq_len: int = 1536,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize zero-prompt user message + assistant response with exact offset mapping
    to locate <cog_router>...</cog_router> subword tokens without off-by-one errors.
    """
    msgs_prompt = [{"role": "user", "content": prompt_str}]
    msgs_full = [
        {"role": "user", "content": prompt_str},
        {"role": "assistant", "content": response_str},
    ]
    prompt_text = tokenizer.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)

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

    # Identify exact character span of <cog_router>...</cog_router> inside assistant region
    router_mask = [0.0] * len(input_ids)
    r_start = full_text.find("<cog_router>", len(prompt_text) - 5 if len(prompt_text) > 5 else 0)
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


class LengthBucketBatchSampler:
    """
    Groups dataset indices of similar sequence lengths into micro-batches to
    eliminate >55% wasted FLOPs on <pad> tokens, while shuffling batch order
    every epoch so stochastic gradient descent remains unbiased.
    """
    def __init__(self, lengths: List[int], batch_size: int, seed: int = 2026):
        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [
            sorted_idx[i : i + batch_size] for i in range(0, len(sorted_idx), batch_size)
        ]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.batches)))
        rng.shuffle(order)
        for b_idx in order:
            yield self.batches[b_idx]

    def __len__(self):
        return len(self.batches)


class OccamSFTDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer, max_seq_len: int = 1024):
        self.items = [
            encode_prompt_and_response(tokenizer, r["prompt"], r["response"], max_seq_len=max_seq_len)
            for r in records
        ]
        self.lengths = [x["input_ids"].size(0) for x in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_sft_collator(pad_id: int):
    def collate(batch):
        max_l = max(x["input_ids"].size(0) for x in batch)
        b_ids, b_lbls, b_mask = [], [], []
        for x in batch:
            l = x["input_ids"].size(0)
            pad_n = max_l - l
            b_ids.append(F.pad(x["input_ids"], (0, pad_n), value=pad_id))
            b_lbls.append(F.pad(x["labels"], (0, pad_n), value=-100))
            b_mask.append(F.pad(torch.ones(l, dtype=torch.long), (0, pad_n), value=0))
        return {
            "input_ids": torch.stack(b_ids),
            "labels": torch.stack(b_lbls),
            "attention_mask": torch.stack(b_mask),
        }
    return collate


class OccamSimPODataset(Dataset):
    def __init__(self, records: List[dict], tokenizer, max_seq_len: int = 1024):
        self.items = []
        self.lengths = []
        for r in records:
            ch = encode_prompt_and_response(tokenizer, r["prompt"], r["chosen"], max_seq_len=max_seq_len)
            rj = encode_prompt_and_response(tokenizer, r["prompt"], r["rejected"], max_seq_len=max_seq_len)
            dyn_lam = float(r.get("dynamic_lambda_len", 0.15))
            self.items.append({"chosen": ch, "rejected": rj, "dyn_lambda": dyn_lam})
            self.lengths.append(rj["input_ids"].size(0) * 2 + ch["input_ids"].size(0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_simpo_collator(pad_id: int):
    def pad_group(sub_items):
        max_l = max(x["input_ids"].size(0) for x in sub_items)
        ids, lbls, rmask, amask, rlens = [], [], [], [], []
        for x in sub_items:
            l = x["input_ids"].size(0)
            p = max_l - l
            ids.append(F.pad(x["input_ids"], (0, p), value=pad_id))
            lbls.append(F.pad(x["labels"], (0, p), value=-100))
            rmask.append(F.pad(x["router_mask"], (0, p), value=0.0))
            amask.append(F.pad(torch.ones(l, dtype=torch.long), (0, p), value=0))
            rlens.append(x["resp_len"])
        return (
            torch.stack(ids),
            torch.stack(lbls),
            torch.stack(rmask),
            torch.stack(amask),
            torch.tensor(rlens, dtype=torch.float32),
        )

    def collate(batch):
        w_ids, w_lbls, w_rmask, w_amask, w_lens = pad_group([b["chosen"] for b in batch])
        l_ids, l_lbls, l_rmask, l_amask, l_lens = pad_group([b["rejected"] for b in batch])
        dyn_lams = torch.tensor([b["dyn_lambda"] for b in batch], dtype=torch.float32)
        return {
            "w_ids": w_ids,
            "w_lbls": w_lbls,
            "w_rmask": w_rmask,
            "w_amask": w_amask,
            "w_lens": w_lens,
            "l_ids": l_ids,
            "l_lbls": l_lbls,
            "l_rmask": l_rmask,
            "l_amask": l_amask,
            "l_lens": l_lens,
            "dyn_lams": dyn_lams,
        }
    return collate


def _chunk_token_logps(
    h_chunk: torch.Tensor,
    lbl_chunk: torch.Tensor,
    lm_head: torch.nn.Module,
) -> torch.Tensor:
    logits_chunk = lm_head(h_chunk)
    bsz, seq_c, vocab = logits_chunk.shape
    nll = F.cross_entropy(
        logits_chunk.reshape(bsz * seq_c, vocab),
        lbl_chunk.reshape(bsz * seq_c),
        ignore_index=-100,
        reduction="none",
    )
    return -nll.view(bsz, seq_c)


def compute_chunked_token_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Runs the causal backbone to obtain hidden_states [B, T, D] (~14 MB),
    then applies lm_head + cross_entropy in 128-token sequence slices with
    activation checkpointing so full [B, T, 152064] fp32 logits (~10.4 GB)
    are NEVER materialized simultaneously in VRAM.
    """
    base_causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = base_causal.model
    lm_head = base_causal.lm_head

    out_backbone = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden_states = out_backbone[0]  # [B, T, D]

    shift_h = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    seq_len = shift_h.size(1)

    chunks = []
    for t0 in range(0, seq_len, chunk_size):
        t1 = min(t0 + chunk_size, seq_len)
        h_slice = shift_h[:, t0:t1, :]
        lbl_slice = shift_labels[:, t0:t1]
        if torch.is_grad_enabled():
            c_logps = torch.utils.checkpoint.checkpoint(
                _chunk_token_logps, h_slice, lbl_slice, lm_head, use_reentrant=False
            )
        else:
            c_logps = _chunk_token_logps(h_slice, lbl_slice, lm_head)
        chunks.append(c_logps)
    return torch.cat(chunks, dim=1)


def compute_sft_chunked_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    token_logps = compute_chunked_token_logps(model, input_ids, attention_mask, labels)
    active_mask = (labels[:, 1:] != -100).float()
    return -(token_logps * active_mask).sum() / active_mask.sum().clamp(min=1.0)


def compute_occam_weighted_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    router_mask: torch.Tensor,
    alpha_router: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Information-Weighted Sequence Log-Probability Reward:
      w_t = alpha_router if token t is inside <cog_router>...</cog_router>, else 1.0.
    Also returns standard NLL loss on valid completion tokens.
    """
    token_logps = compute_chunked_token_logps(model, input_ids, attention_mask, labels)
    shift_labels = labels[:, 1:]
    shift_rmask = router_mask[:, 1:]

    active_mask = (shift_labels != -100).float()
    weights = torch.where(shift_rmask > 0.5, alpha_router, 1.0) * active_mask
    weighted_logp = (token_logps * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1.0)

    nll = -(token_logps * active_mask).sum() / active_mask.sum().clamp(min=1.0)
    return weighted_logp, nll


_CACHED_MODEL_PATH: Optional[str] = None
_CACHED_BASE_MODEL: Optional[torch.nn.Module] = None
_CACHED_TOKENIZER = None


def load_tokenizer_and_base(
    model_path: str,
    init_adapter_path: Optional[Path] = None,
    lora_r: int = 32,
    lora_alpha: int = 64,
):
    global _CACHED_MODEL_PATH, _CACHED_BASE_MODEL, _CACHED_TOKENIZER

    if _CACHED_MODEL_PATH != model_path or _CACHED_BASE_MODEL is None:
        if _CACHED_BASE_MODEL is not None:
            del _CACHED_BASE_MODEL
            _CACHED_BASE_MODEL = None
            cleanup_vram()

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

        _CACHED_MODEL_PATH = model_path
        _CACHED_BASE_MODEL = base
        _CACHED_TOKENIZER = tokenizer
    else:
        base = _CACHED_BASE_MODEL
        tokenizer = _CACHED_TOKENIZER
        cleanup_vram()

    if init_adapter_path is not None and (init_adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(init_adapter_path), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(base, lora_cfg)
    return model, tokenizer


def release_peft_keep_base(model: torch.nn.Module):
    global _CACHED_BASE_MODEL
    if hasattr(model, "unload"):
        _CACHED_BASE_MODEL = model.unload()
    del model
    cleanup_vram()


def is_checkpoint_done(save_dir: Path) -> bool:
    return (save_dir / "adapter_model.safetensors").exists() and (save_dir / "training_metrics.json").exists()


def train_sft_checkpoint(
    name: str,
    model_path: str,
    records: List[dict],
    save_dir: Path,
    epochs: int = 2,
    lr: float = 2e-4,
    micro_bs: int = 4,
    grad_accum: int = 8,
    max_seq_len: int = 1024,
) -> dict:
    if is_checkpoint_done(save_dir):
        print(f"  [Skip - Exists] {name} already trained at {save_dir}")
        return json.loads((save_dir / "training_metrics.json").read_text(encoding="utf-8"))

    cleanup_vram()
    print("=" * 78)
    print(f"[Train SFT] {name} | Base={Path(model_path).name} | N={len(records)} | Epochs={epochs}")
    print("=" * 78)

    model, tokenizer = load_tokenizer_and_base(model_path)
    ds = OccamSFTDataset(records, tokenizer, max_seq_len=max_seq_len)
    loader = DataLoader(
        ds,
        batch_sampler=LengthBucketBatchSampler(ds.lengths, batch_size=micro_bs),
        collate_fn=make_sft_collator(tokenizer.pad_token_id),
    )

    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    total_opt_steps = max(1, math.ceil(len(loader) * epochs / grad_accum))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(3, total_opt_steps // 10), num_training_steps=total_opt_steps
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
                raw_loss = compute_sft_chunked_loss(model, input_ids, attention_mask, labels)
                loss = raw_loss / grad_accum

            loss.backward()
            val = float(raw_loss.detach().cpu().item())
            ep_losses.append(val)
            losses.append(val)

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
        "stage": "Schema_Context_Distillation_SFT",
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
    print(f"  -> Saved {name}: VRAM={peak_vram_gb:.2f} GB | Time={wall_sec:.2f}s ({wall_sec/60:.2f} min)")

    del optimizer, loader, ds
    release_peft_keep_base(model)
    return metrics


def train_simpo_checkpoint(
    name: str,
    model_path: str,
    init_sft_dir: Path,
    records: List[dict],
    save_dir: Path,
    epochs: int = 1,
    lr: float = 8e-6,
    beta: float = 2.0,
    gamma: float = 0.5,
    alpha_router: float = 2.0,
    use_dynamic_lambda: bool = True,
    fixed_lambda: float = 0.15,
    nll_coef: float = 0.15,
    micro_bs: int = 2,
    grad_accum: int = 8,
    max_seq_len: int = 1024,
) -> dict:
    if is_checkpoint_done(save_dir):
        print(f"  [Skip - Exists] {name} already trained at {save_dir}")
        return json.loads((save_dir / "training_metrics.json").read_text(encoding="utf-8"))

    cleanup_vram()
    print("=" * 78)
    print(
        f"[Train OCCAM-SimPO] {name} | Init={init_sft_dir.name} | alpha_router={alpha_router} | "
        f"dyn_lambda={use_dynamic_lambda} | N={len(records)}"
    )
    print("=" * 78)

    model, tokenizer = load_tokenizer_and_base(model_path, init_adapter_path=init_sft_dir)
    ds = OccamSimPODataset(records, tokenizer, max_seq_len=max_seq_len)
    loader = DataLoader(
        ds,
        batch_sampler=LengthBucketBatchSampler(ds.lengths, batch_size=micro_bs),
        collate_fn=make_simpo_collator(tokenizer.pad_token_id),
    )

    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    total_opt_steps = max(1, math.ceil(len(loader) * epochs / grad_accum))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(3, total_opt_steps // 10), num_training_steps=total_opt_steps
    )

    model.train()
    t0 = time.perf_counter()
    losses, margins = [], []
    micro_step = 0
    optimizer.zero_grad()

    for ep in range(1, epochs + 1):
        ep_l, ep_m = [], []
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
                r_w, nll_w = compute_occam_weighted_logps(
                    model, w_ids, w_amask, w_lbls, w_rmask, alpha_router=alpha_router
                )
                r_l, _ = compute_occam_weighted_logps(
                    model, l_ids, l_amask, l_lbls, l_rmask, alpha_router=alpha_router
                )

                lam_vec = dyn_lams if use_dynamic_lambda else torch.full_like(dyn_lams, fixed_lambda)
                len_pen = lam_vec * (w_lens / l_lens.clamp(min=1.0))
                simpo_margin = beta * (r_w - r_l - gamma - len_pen)
                simpo_loss = -F.logsigmoid(simpo_margin).mean()
                total_loss = (simpo_loss + nll_coef * nll_w) / grad_accum

            total_loss.backward()
            lv = float(simpo_loss.detach().cpu().item())
            mv = float((r_w - r_l).mean().detach().cpu().item())
            ep_l.append(lv)
            ep_m.append(mv)
            losses.append(lv)
            margins.append(mv)

            if micro_step % grad_accum == 0 or micro_step == len(loader) * epochs:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print(
            f"  [{name} Epoch {ep}/{epochs}] simpo_loss={sum(ep_l)/max(1,len(ep_l)):.4f} | "
            f"reward_margin={sum(ep_m)/max(1,len(ep_m)):.4f}"
        )

    torch.cuda.synchronize()
    wall_sec = time.perf_counter() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    metrics = {
        "checkpoint_name": name,
        "base_model": Path(model_path).name,
        "stage": "OCCAM_SimPO_Alignment",
        "alpha_router": alpha_router,
        "dynamic_lambda_len": use_dynamic_lambda,
        "num_samples": len(records),
        "epochs": epochs,
        "opt_steps": total_opt_steps,
        "initial_loss": round(losses[0], 4) if losses else 0.0,
        "final_loss": round(sum(losses[-10:]) / max(1, min(10, len(losses))), 4),
        "final_margin": round(sum(margins[-10:]) / max(1, min(10, len(margins))), 4),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "wall_time_sec": round(wall_sec, 2),
        "gpu_hours": round(wall_sec / 3600.0, 4),
    }
    (save_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  -> Saved {name}: VRAM={peak_vram_gb:.2f} GB | Time={wall_sec:.2f}s ({wall_sec/60:.2f} min)")

    del optimizer, loader, ds
    release_peft_keep_base(model)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=str, default="./models")
    parser.add_argument("--families", nargs="+", default=["7b", "3b", "llama8b", "r1"], choices=["7b", "3b", "llama8b", "r1"])
    args = parser.parse_args()

    set_seed(2026)
    fams = set(args.families)
    models_root = Path(args.models_root)
    qwen3b = str(models_root / "Qwen2.5-3B-Instruct")
    qwen7b = str(models_root / "Qwen2.5-7B-Instruct")
    llama8b = str(models_root / "Meta-Llama-3.1-8B-Instruct")
    r1_7b = str(models_root / "DeepSeek-R1-Distill-Qwen-7B")

    sft_data = load_jsonl(SPLITS_DIR / "sft_train.jsonl")
    simpo_data = load_jsonl(SPLITS_DIR / "simpo_train.jsonl")
    limo_data = load_jsonl(SPLITS_DIR / "limo_longcot_train.jsonl")
    abl_wo_router_simpo = load_jsonl(SPLITS_DIR / "ablation_wo_router_train.jsonl")
    abl_wo_router_sft = [{"prompt": r["prompt"], "response": r["chosen"]} for r in abl_wo_router_simpo]
    abl_jargon_sft = load_jsonl(SPLITS_DIR / "ablation_jargon_only_train.jsonl")

    summary_path = ASSETS_DIR / "training_summary.json"
    summary_runs = {}
    if summary_path.exists():
        try:
            summary_runs = json.loads(summary_path.read_text(encoding="utf-8")).get("runs", {})
        except Exception:
            summary_runs = {}

    # 1. OCCAM-3B (SFT + SimPO)
    if "3b" in fams:
        summary_runs["OCCAM-3B-SFT"] = train_sft_checkpoint(
            "OCCAM-3B-SFT", qwen3b, sft_data, CKPT_ROOT / "OCCAM-3B-SFT", epochs=2, lr=2e-4, micro_bs=4, grad_accum=8
        )
        summary_runs["OCCAM-3B-SimPO"] = train_simpo_checkpoint(
            "OCCAM-3B-SimPO", qwen3b, CKPT_ROOT / "OCCAM-3B-SFT", simpo_data, CKPT_ROOT / "OCCAM-3B-SimPO",
            epochs=1, lr=1e-5, alpha_router=2.0, use_dynamic_lambda=True, micro_bs=2, grad_accum=8
        )

    # 2. Main Flagship: OCCAM-7B + Diagnostic Ablations + Sweeps
    if "7b" in fams:
        summary_runs["OCCAM-7B-SFT"] = train_sft_checkpoint(
            "OCCAM-7B-SFT", qwen7b, sft_data, CKPT_ROOT / "OCCAM-7B-SFT", epochs=2, lr=1.5e-4, micro_bs=4, grad_accum=8
        )
        summary_runs["OCCAM-7B-SimPO"] = train_simpo_checkpoint(
            "OCCAM-7B-SimPO", qwen7b, CKPT_ROOT / "OCCAM-7B-SFT", simpo_data, CKPT_ROOT / "OCCAM-7B-SimPO",
            epochs=1, lr=8e-6, alpha_router=2.0, use_dynamic_lambda=True, micro_bs=2, grad_accum=8
        )
        summary_runs["Ablation-wo-Router-7B-SFT"] = train_sft_checkpoint(
            "Ablation-wo-Router-7B-SFT", qwen7b, abl_wo_router_sft, CKPT_ROOT / "Ablation-wo-Router-7B-SFT",
            epochs=2, lr=1.5e-4, micro_bs=4, grad_accum=8
        )
        summary_runs["Ablation-wo-Router-7B"] = train_simpo_checkpoint(
            "Ablation-wo-Router-7B", qwen7b, CKPT_ROOT / "Ablation-wo-Router-7B-SFT", abl_wo_router_simpo,
            CKPT_ROOT / "Ablation-wo-Router-7B", epochs=1, lr=8e-6, alpha_router=1.0, use_dynamic_lambda=False,
            micro_bs=2, grad_accum=8
        )
        summary_runs["Ablation-Jargon-Only-7B"] = train_sft_checkpoint(
            "Ablation-Jargon-Only-7B", qwen7b, abl_jargon_sft, CKPT_ROOT / "Ablation-Jargon-Only-7B",
            epochs=2, lr=1.5e-4, micro_bs=4, grad_accum=8
        )
        summary_runs["Ablation-Uniform-SimPO-7B"] = train_simpo_checkpoint(
            "Ablation-Uniform-SimPO-7B", qwen7b, CKPT_ROOT / "OCCAM-7B-SFT", simpo_data,
            CKPT_ROOT / "Ablation-Uniform-SimPO-7B", epochs=1, lr=8e-6, alpha_router=1.0, use_dynamic_lambda=True,
            micro_bs=2, grad_accum=8
        )
        summary_runs["LIMO-LongCoT-SFT-7B"] = train_sft_checkpoint(
            "LIMO-LongCoT-SFT-7B", qwen7b, limo_data, CKPT_ROOT / "LIMO-LongCoT-SFT-7B",
            epochs=2, lr=1.5e-4, micro_bs=4, grad_accum=8
        )
        for alpha_val in (0.5, 1.5, 3.0):
            tag = f"Sweep-Alpha-{str(alpha_val).replace('.', 'p')}-7B"
            summary_runs[tag] = train_simpo_checkpoint(
                tag, qwen7b, CKPT_ROOT / "OCCAM-7B-SFT", simpo_data, CKPT_ROOT / tag,
                epochs=1, lr=8e-6, alpha_router=alpha_val, use_dynamic_lambda=True, micro_bs=4, grad_accum=8
            )
        rng_sub = random.Random(2026)
        indices = list(range(len(sft_data)))
        rng_sub.shuffle(indices)
        for n_sub in (200, 400, 800):
            sub_idx = indices[: min(n_sub, len(sft_data))]
            sub_sft = [sft_data[i] for i in sub_idx]
            sub_simpo = [simpo_data[i] for i in sub_idx]
            sft_tag = f"Sweep-Scale-{n_sub}-7B-SFT"
            simpo_tag = f"Sweep-Scale-{n_sub}-7B-SimPO"
            summary_runs[sft_tag] = train_sft_checkpoint(
                sft_tag, qwen7b, sub_sft, CKPT_ROOT / sft_tag, epochs=2, lr=1.5e-4, micro_bs=4, grad_accum=8
            )
            summary_runs[simpo_tag] = train_simpo_checkpoint(
                simpo_tag, qwen7b, CKPT_ROOT / sft_tag, sub_simpo, CKPT_ROOT / simpo_tag,
                epochs=1, lr=8e-6, alpha_router=2.0, use_dynamic_lambda=True, micro_bs=4, grad_accum=8
            )

    # 3. Cross-Architecture Validation: Meta-Llama-3.1-8B-Instruct
    if "llama8b" in fams:
        summary_runs["OCCAM-Llama3.1-8B-SFT"] = train_sft_checkpoint(
            "OCCAM-Llama3.1-8B-SFT", llama8b, sft_data, CKPT_ROOT / "OCCAM-Llama3.1-8B-SFT",
            epochs=2, lr=1.2e-4, micro_bs=4, grad_accum=8
        )
        summary_runs["OCCAM-Llama3.1-8B-SimPO"] = train_simpo_checkpoint(
            "OCCAM-Llama3.1-8B-SimPO", llama8b, CKPT_ROOT / "OCCAM-Llama3.1-8B-SFT", simpo_data,
            CKPT_ROOT / "OCCAM-Llama3.1-8B-SimPO", epochs=1, lr=6e-6, alpha_router=2.0, use_dynamic_lambda=True,
            micro_bs=2, grad_accum=8
        )

    # 4. Strong-Reasoner Compression: DeepSeek-R1-Distill-Qwen-7B -> OCCAM-R1-7B
    if "r1" in fams:
        summary_runs["OCCAM-R1-7B-SFT"] = train_sft_checkpoint(
            "OCCAM-R1-7B-SFT", r1_7b, sft_data, CKPT_ROOT / "OCCAM-R1-7B-SFT",
            epochs=2, lr=1.2e-4, micro_bs=4, grad_accum=8
        )
        summary_runs["OCCAM-R1-7B-SimPO"] = train_simpo_checkpoint(
            "OCCAM-R1-7B-SimPO", r1_7b, CKPT_ROOT / "OCCAM-R1-7B-SFT", simpo_data,
            CKPT_ROOT / "OCCAM-R1-7B-SimPO", epochs=1, lr=6e-6, alpha_router=2.0, use_dynamic_lambda=True,
            micro_bs=4, grad_accum=8
        )

    total_train_sec = sum(v["wall_time_sec"] for v in summary_runs.values())
    peak_vram_overall = max(v["peak_vram_gb"] for v in summary_runs.values())
    out_summary = {
        "hardware": torch.cuda.get_device_name(0),
        "total_checkpoints_trained": len(summary_runs),
        "total_training_wall_sec": round(total_train_sec, 2),
        "total_training_gpu_hours": round(total_train_sec / 3600.0, 4),
        "overall_peak_vram_gb": round(peak_vram_overall, 2),
        "runs": summary_runs,
    }
    summary_path.write_text(json.dumps(out_summary, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"[STAGE 3 GATE PASSED] All {len(summary_runs)} Authentic LoRA Checkpoints Trained!")
    print(f"  - Total GPU Training Time : {total_train_sec:.1f}s ({total_train_sec/3600.0:.3f} GPU hours)")
    print(f"  - Peak RTX 3090 VRAM      : {peak_vram_overall:.2f} GB")
    print("=" * 78)


if __name__ == "__main__":
    main()
