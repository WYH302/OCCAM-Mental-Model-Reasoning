# OCCAM: Teaching Language Models to Think in Mental Models, Not Endless Chains
### 让语言模型用结构化心智模型思考，而非冗长无休止的思维链

<p align="center">
  <a href="#english-overview"><b>English Overview</b></a> •
  <a href="#engineering-reflections-pitfalls--observations"><b>Pitfalls &amp; Notes</b></a> •
  <a href="#中文简介"><b>中文简介</b></a> •
  <a href="#dataset-occam-2k--数据集说明"><b>OCCAM-2K Dataset</b></a> •
  <a href="#quickstart--快速复现"><b>Quickstart</b></a>
</p>

<p align="center">
  <img src="assets/fig0_framework_architecture.png" width="100%" alt="OCCAM Overall Framework"/>
</p>

---

## English Overview

This repository shares the research code, evaluation suite, and the **OCCAM-2K** dataset for studying **compact mental-model reasoning** in open-weight small and medium-sized language models (`3B`–`7B`) under a single consumer `24 GB` GPU budget (`NVIDIA RTX 3090`), without relying on external commercial APIs.

### Motivation & Observations

Extended chain-of-thought (Long-CoT) generation substantially helps multi-step symbolic search, yet on many causal and logical framing problems, unconstrained models tend to spend hundreds or thousands of tokens in repetitive self-verification loops while still getting tripped up by confounders or deductive traps.

In this exploratory study, we investigate a simple question: **Can compile-time guidance from reusable cognitive primitives (mental models) be distilled into compact, zero-prompt student policies?**

1. **Framing-vs-Rollout Rate-Distortion Perspective**: On *structural framing* benchmarks (`CLadder`, `BBH-Hard`, `ARC-Challenge`, `LogiQA`), paying fewer than 12 bits of upfront routing inside `<cog_router>` prunes redundant search branches and lowers downstream token conditional entropy (`-74.4%` in our 250-trajectory probe). Conversely, multi-step competition algebra (`MATH-500`) exhibits an expected rate-distortion trade-off when step budgets are tightly compressed.
2. **Pure Local Self-Distillation (`OCCAM-2K`)**: Using `Qwen2.5-7B-Instruct` locally with deterministic `SymPy` and logic verification (`0.00%` SHA-256 and 13-gram train-test overlap across `4,526` official test problems), we curate `1,570` verified structured preference pairs (`232.7` chosen tokens vs. `544.1` unstructured tokens) alongside `1,345` matched-source Long-CoT controls.
3. **Router-Weighted Alignment (`OCCAM-SimPO`) & Memory-Efficient LoRA**: Because `<cog_router>` occupies only ~14% of sequence tokens, upweighting router tokens (`alpha_router > 1`) mitigates prefix gradient dilution on unseen out-of-distribution tasks, while 256-token activation-checkpointed `lm_head` slicing keeps `bf16` SimPO training comfortably within a single `24 GB` GPU (`7.51 GB` peak VRAM on `3B`, `16.78 GB` on `7B`).

<p align="center">
  <img src="assets/fig1_dichotomy_overview.png" width="100%" alt="Structural Framing vs Rollout Profile"/>
</p>

### Summary of Empirical Findings (`N = 4,526` Official Test Items Across 8 Benchmarks)

All evaluations were conducted locally on one `RTX 3090 (24 GB)` GPU with deterministic `SymPy`/logic grading (`0.00%` 13-gram train–test overlap). For adaptive configurations, Phase-1 router tokens and Phase-2 execution tokens/latency are strictly summed.

| Model / Paradigm | Framing-4 (`N=2,723`) | ID Overall (`N=3,223`) | MATH-500 (`N=500`) | All-8 Micro (`N=4,526`) | Avg ID Tokens | Dynamic `bf16` KV | Speedup vs. R1-7B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen2.5-3B-Instruct` (Zero-Shot CoT) | 52.95% | 59.88% | 65.0% | 64.74% | 360.3 | 12.7 MB | 13.5x |
| **`OCCAM-3B-SFT` (Zero-Prompt)** | **58.37%** | 62.43% | 57.2% | 64.49% | **257.3** | **9.1 MB** | **14.5x** |
| **`OCCAM-3B-Adaptive` (3-Seed Mean ± Std)** | 57.20±0.08% | **62.68±0.14%** | 62.7% | **66.40±0.16%** | 304.1±2.1 | 10.7 MB | 11.4x |
| `Chain-of-Draft / Concise-CoT` (7B) | 60.82% | 62.40% | 36.4% | 61.16% | 46.9 | 2.6 MB | 26.8x |
| `Buffer of Thoughts RAG` (7B, Top-2) | 63.59% | 70.40% | 73.4% | 73.09% | 429.5 | 23.5 MB | 4.9x |
| `LIMO Long-CoT SFT` (COLM '25, 7B) | 67.10% | 72.17% | 72.6% | 75.01% | 504.9 | 27.6 MB | 4.2x |
| `TokenSkip-50 SFT` (EMNLP '25, 7B) | 62.90% | 68.94% | 69.8% | 71.48% | 428.5 | 23.4 MB | 3.9x |
| `R1-NoThink Bypass` (NeurIPS '25, 7B) | 46.60% | 52.68% | 60.0% | 58.35% | 251.7 | 13.8 MB | 7.9x |
| `R1-AdaptThink Gate` (EMNLP '25, 7B) | 49.60% | 58.61% | 87.2% | 63.57% | 659.6 | 36.1 MB | 2.4x |
| `DeepSeek-R1-Distill-Qwen-7B` (Budget 200t) | 39.93% | 39.71% | 27.4% | 45.10% | 185.4 | 10.1 MB | 1.4x |
| `DeepSeek-R1-Distill-Qwen-7B` (8k CoT) | 64.99% | 70.56% | **87.6%** | 74.04% | 1,457.8 | 79.7 MB | 1.0x |
| `Qwen2.5-7B-Instruct` (Zero-Shot CoT) | 65.48% | 71.86% | 74.6% | 75.06% | 329.3 | 18.0 MB | 6.9x |
| `Qwen2.5-7B-Instruct` (`SC maj@4`) | 66.77% | 73.22% | 76.2% | 76.27% | 1,307.6 | 71.5 MB | 2.0x |
| `Ext-RouteLLM-7B` (ICLR '25, OCCAM-Tax.) | 68.47% | 75.55% | **87.6%** | **78.08%** | 795.0 | 43.5 MB | 1.9x |
| `Abl. FFN-Frozen AttnLoRA` (`0.13%` params) | 64.29% | 70.00% | 69.4% | 72.38% | 262.9 | 14.4 MB | 5.5x |
| **`OCCAM-Qwen-7B-SimPO` (Calibrated `K=25`)** | 65.91% | 70.09% | 63.0% | 71.76% | **220.2** (`-84.9%`) | **12.0 MB** | **10.0x** |
| **`OCCAM-Qwen-7B (Direct, Zero-Prompt)`** | 67.10% (`57.1%` LogiQA) | 71.67% | 68.6% | 73.35% | 262.5 (`-82.0%`) | 14.4 MB | 9.3x |
| **`OCCAM-Qwen-7B-Adaptive` (Single-7B Gated)** | 66.62% | 71.89% | 73.0% | 74.17% | 294.0 (`-79.8%`) | 16.1 MB | 6.5x |
| **`OCCAM-Router-7B` (Native `<cog_router>`)** | 67.90% | 74.84% | **87.4%** | 76.38% | **699.1** (`-52.0%`) | **38.2 MB** | **2.0x** |
| **`OCCAM-Router-7B+` (WiSE γ=0.65 + Gate)** | **68.83%** | **75.71%** (`+5.15%` vs. R1) | **87.4%** | **77.33%** | 755.7 (`-48.2%`) | 41.3 MB | **2.0x** |

<p align="center">
  <img src="assets/fig2_pareto_frontier.png" width="100%" alt="Accuracy-Token Pareto Frontier and KV-Cache Footprint"/>
</p>

<p align="center">
  <img src="assets/fig6_radar_and_length_dist.png" width="100%" alt="Holistic 6-Axis Capability Radar and Per-Query Token Distribution"/>
</p>

<p align="center">
  <img src="assets/fig3_mechanistic_probes.png" width="100%" alt="Mechanistic Entropy Collapse and Router Intervention"/>
</p>

<p align="center">
  <img src="assets/fig7_qualitative_comparison.png" width="100%" alt="Side-by-Side SOTA Qualitative Comparison with Ground Truth"/>
</p>

### Engineering Reflections, Pitfalls & Observations

Building and verifying all 47 configurations (`212,722` trajectories, `31` LoRA adapters) on a single `24 GB` RTX 3090 surfaced several practical failure modes that rarely appear in polished papers:

1. **Bit-Aligning Native Thinking Templates Matters More Than Expected**: When adapting `DeepSeek-R1-Distill-Qwen-7B`, our first SFT run trained targets outside its native `<think>...</think>` boundary and collapsed accuracy by `-13.2%`. Aligning `<cog_router>` and `<cog_trace>` strictly inside the backbone's pre-existing `<think>` header resolved the format clash immediately.
2. **Why Reference-Free SimPO Peaks Early (`K=25` vs. `K=99`)**: Standard SimPO averages log-probabilities uniformly across all tokens, diluting gradients on the short (~14% length) `<cog_router>` prefix (`1:6` dilution). Upweighting router tokens (`alpha_r > 1`) recovered up to `+2.93%` unseen OOD accuracy, while sweeping training steps revealed that pushing past `K=25` triggers likelihood displacement on out-of-distribution suites.
3. **Diagnosing FFN Factual Drift on `StrategyQA` (`0.13%` Attention-LoRA → Zero-Cost WiSE-FT)**: Full-MLP LoRA (`0.53%` params) pushed `LogiQA` to `#1` (`57.1%`) but nudged zero-shot `StrategyQA` down (`67.3% → 56.3%`). Mixing `380` replay traces only partially helped and weakened `MATH` routing (`99.0% → 74.2%`). Freezing all FFN layers (`q,k,v,o_proj` only, `0.13%` params) immediately restored `+4.3%` on `StrategyQA`, empirically confirming Geva et al.'s key-value FFN memory hypothesis. Combining weight-space interpolation (`WiSE-FT`, `γ=0.65`) with a `3.5K`-parameter prefill linear head then unified both regimes with `0` extra GPU training seconds (`90.8%` ARC-C, `76.4%` FOLIO, `75.10%` OOD, `77.33%` All-8).
4. **Fitting `152,064`-Vocab `bf16` SimPO on One `24 GB` Card**: Materializing full `[B, L, 152064]` logit tensors OOMs a `24 GB` GPU during paired preference optimization. Computing cross-entropy over `lm_head` in `256`-token activation-checkpointed slices cut peak SimPO VRAM to `7.51 GB` (`3B`) and `16.78 GB` (`7B`) with zero mathematical approximation.

---

## 中文简介

本项目开源了 **OCCAM**（基于结构化心智模型对齐的认知推理压缩）的实验代码、评测脚本与 **OCCAM-2K** 数据集。我们希望在**单张消费级显卡（24 GB 显存，零商业大模型 API 依赖）**条件下，踏实探讨如何将有明确数学/诺奖/图灵奖理论与脑科学机制背书的 16 个正交思维原语（因果图干预、波普尔证伪、逆向目标归约、博弈非对称匹配等）内化到 `3B`–`7B` 开源模型中，实现结构建构任务与符号展开任务的自适应路由与高效压缩。

### 主要工作与实测观察

1. **区分“结构建构型任务”与“符号展开型任务”的率失真边界与零提示自适应路由**：
   - 在因果推断（`CLadder`）、演绎逻辑（`LogiQA`）、复合推理（`BBH-Hard`）与科学推理（`ARC-Challenge`）等结构建构任务上，通过 `<cog_router>` 付出不到 12 比特的前置结构选择代价，即可显著裁剪后续无效搜索分支（在 250 条测试轨迹探针中，下游 Token 条件香农熵下降 `74.4%`）。
   - 在 7B 激活参数预算下，`OCCAM-Router-7B+`（WiSE-FT `γ=0.65` 权重插值 + 3.5K 预填充线性门控）与原生零提示 `OCCAM-Router-7B`（`100.0%` XML 合规）在全部 `4,526` 道官方去污测试题上分别取得 **`75.71%` / `74.84%` 分布内总准确率**（超越 8k `DeepSeek-R1-Distill-7B` `+5.15%`，超越 EMNLP '25 `TokenSkip-50` `+6.77%`，超越 EMNLP '25 `R1-AdaptThink` `+17.10%`）、**`68.83%` / `67.90%` 结构建构均分** 与 **`77.33%` / `76.38%` 八基准总准确率**，同时较 R1-7B 减少 `1.93×–2.09×` Token；单骨干 `OCCAM-3B-Adaptive` 在 3 组独立随机种子（`{2026, 42, 1337}`）上实现 `62.68 ± 0.14%` ID 与 `66.40 ± 0.16%` All-8 极低方差复现。
2. **纯本地自蒸馏与严格去污的数据集 `OCCAM-2K`**：
   - 全流程仅使用本地开源模型配合 `SymPy` 符号与逻辑确定性校验，从 1,600 道种子题中合成 1,570 对结构化偏好样本（正样本平均 `232.7` tokens，无结构负样本平均 `544.1` tokens），并与全部 `4,526` 道官方测试题保持 `0.00%` SHA-256 及 13-gram 重合。
3. **显式路由加权对齐（`OCCAM-SimPO`）与单卡显存优化**：
   - 针对简短路由前缀梯度易被长序列稀释的问题，对 `<cog_router>` 赋予更高信用权重，并在无参考模型对齐中结合 256-Token 词表分块激活重计算技术，使 `152k` 大词表模型的训练显存保持在 `7.5–17.8 GB` 以内。

### 真实思考过程、踩坑记录与工程观察

在单张 `RTX 3090 (24 GB)` 上跑通 31 个 LoRA 权重与 47 组全量评测的过程中，我们遇到了几个很有启发性的真实“深坑”，特此整理供复现与后续研究参考：

- **坑一：忽视推理模型原生 `<think>` 模板边界会直接掉点 `-13.2%`**。早期微调 `DeepSeek-R1-Distill-7B` 时，我们将结构化标签直接写在原生 `<think>` 之外，导致模型模板分布冲突；将 `<cog_router>` 与 `<cog_trace>` 逐字节对齐放回 `<think>...</think>` 内部后，路由与推理能力才恢复正常。
- **坑二：无参考偏好优化（SimPO）并非步数越多越好**。由于 `<cog_router>` 仅占全序列约 `14%` 的长度，均匀平均会严重稀释路由梯度；加入路由加权（`alpha_r > 1`）后 OOD 精度提升最高 `+2.93%`，同时扫步发现 `K=25` 是最优压缩拐点（`220.2` tokens、`10.0×` 提速），继续拉长到 `K=99` 反而会触发似然位移（Likelihood Displacement）导致泛化回落。
- **坑三：从 `StrategyQA` 常识漂移中验证 FFN 键值记忆假说，并用零成本 WiSE-FT 破局**。全 MLP LoRA 虽然把 `LogiQA` 拉到了全榜第一（`57.1%`），却让零样本百科常识 `StrategyQA` 从 `67.3%` 降到 `56.3%`。我们没有盲目调参，而是做了三路消融：加入 `380` 条自回放数据虽略有回升但稀释了数学路由；**冻结全部 FFN 层、仅微调注意力 `q,k,v,o_proj`（仅 `0.13%` 参数）** 后 `StrategyQA` 立刻回弹 `+4.3%`，直接验证了 FFN 作为事实记忆体的机理；最终通过零额外训练成本的 **WiSE-FT 权重插值（`γ=0.65`）+ 3.5K 预填充线性门控**，在零增训开销下将 `ARC-Challenge` 推至 `90.8%`、`FOLIO` 推至 `76.4%`、All-8 推至 `77.33%`。
- **坑四：单卡 `24 GB` 训 `152,064` 大词表 SimPO 的显存墙**。原生计算整段 Logits 矩阵会直接触发 OOM；通过将 `lm_head` 交叉熵切分为 `256`-Token 激活检查点分块计算，在严格零数学近似的前提下将峰值显存压至 `7.51 GB (3B)` / `16.78 GB (7B)`，并在 Windows WDDM 下配合分段批量推理防止 TDR 超时。

<p align="center">
  <img src="assets/fig5_ablation_and_horizon.png" width="100%" alt="Ablation Studies and Preference Optimization Horizon"/>
</p>

<p align="center">
  <img src="assets/fig4_cognitive_heatmap.png" width="100%" alt="Tag-Stripped Blind Semantic Probe Heatmap"/>
</p>

---

## Dataset (`OCCAM-2K`) / 数据集说明

All decontaminated splits are included directly in [`dataset/OCCAM-2K/`](dataset/OCCAM-2K/) in standard Parquet format (`~3.3 MB` total):
- `train-00000-of-00001.parquet`: `1,570` verified `(question, y_chosen, y_rejected, activated_primitives, compression_regime)` pairs.
- `test_id-00000-of-00001.parquet`: `3,223` official in-distribution test problems (`CLadder`, `BBH-Hard`, `ARC-Challenge`, `LogiQA`, `MATH-500`).
- `test_ood-00000-of-00001.parquet`: `1,303` zero-shot out-of-distribution test problems (`GSM8K`, `FOLIO`, `StrategyQA`).

---

## Repository Structure / 目录结构

```text
├── assets/                  # Framework & experimental visualization figures
├── configs/
│   └── cognitive_primitives_16.json   # Formal specification of 16 cognitive primitives
├── dataset/
│   └── OCCAM-2K/            # Decontaminated Parquet train / ID-test / OOD-test splits
├── results/                 # Retained summary statistics, training logs & probe metrics
├── scripts/
│   ├── 00_validate_primitives.py      # Schema validation gate
│   ├── 01_fetch_and_sample.py         # Benchmark sampling & 13-gram decontamination gate
│   ├── 02_harness_synthesizer.py      # Local vLLM self-distillation & SymPy verification
│   ├── 02b_refine_negatives_and_regimes.py
│   ├── 03_train_cog_simpo.py          # Chunked-lm_head LoRA SFT & OCCAM-SimPO trainer
│   ├── 04a_vllm_eval_runner.py        # Multi-LoRA vLLM 8-benchmark evaluation & probes
│   ├── 04b_stats_probes_and_plots.py  # Bootstrap CIs, McNemar tests, blind probe & plots
│   └── utils_grader.py                # Format-agnostic symbolic equivalence grader
├── requirements.txt
└── LICENSE
```

---

## Quickstart / 快速复现

```bash
# 1. Install dependencies (Python 3.10+, CUDA 12.x/13.x recommended for vLLM)
pip install -r requirements.txt

# 2. Validate cognitive primitive schema
python scripts/00_validate_primitives.py

# 3. Train LoRA SFT & OCCAM-SimPO checkpoints on a single 24GB GPU
python scripts/03_train_cog_simpo.py --models-root /path/to/models --families 3b 7b

# 4. Run 8-benchmark vLLM evaluation & mechanistic probes
python scripts/04a_vllm_eval_runner.py --models-root /path/to/models --groups A B C

# 5. Recompute statistical significance & regenerate plots
python scripts/04b_stats_probes_and_plots.py
```

---

## Acknowledgments & Licenses / 致谢与开源协议

- Code in this repository is released under the **MIT License**.
- The **OCCAM-2K** dataset is shared under **CC-BY-4.0**, respecting the upstream open-source licenses of `CLadder`, `BIG-Bench Hard`, `AI2 ARC`, `LogiQA`, `MATH`, `GSM8K`, `FOLIO`, `StrategyQA`, `Qwen2.5`, and `DeepSeek-R1-Distill`.
- Feedback, issues, and constructive discussions are very welcome. 欢迎交流指正与讨论。
