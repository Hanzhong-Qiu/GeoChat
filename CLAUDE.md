# GeoChat Soft Labeling Project — Context Handover

> **Last updated:** 2026-04-23
> **Author:** Han (`dw22963`), Final-Year CS, University of Bristol
> **Purpose:** Single source of truth for this dissertation project. Read this first before responding to any project-related question.

---

## 0. How to Use This Document

如果你是 Claude Code 在 Isambard 上读取这份文档:
- Section 1–6 是项目本身,基本不变
- Section 7–9 是当前进度和环境,会经常更新
- Section 10–12 是写作风格和我的偏好,严格遵循

---

## 1. Project Overview / 项目核心

**Working title:** *Distance-Aware Soft Labeling for Remote Sensing Visual Grounding via GeoChat*

**Research question:**
Can replacing standard cross-entropy hard labeling with a distance-aware soft labeling loss on bounding-box coordinate digit tokens improve visual grounding accuracy in remote sensing MLLMs (specifically GeoChat)?

**Contribution (claimed):**
1. First application of digit-token soft labeling (Wang et al. ICCV 2025) to **remote sensing visual grounding** (paper原本是 general numerical prediction)
2. Controlled comparison: hard-label vs soft-label LoRA fine-tuned GeoChat variants on VRSBench, both starting from the same `geochat-7B` base
3. Empirical & theoretical analysis of why per-digit soft labeling is fundamentally limited for multi-digit coordinate prediction (这点是 negative finding, 但有学术价值)

**Theoretical basis:**
Wang et al., *Enhancing Numerical Prediction of MLLMs with Soft Labeling*, ICCV 2025.

---

## 2. Background Knowledge

### 2.1 GeoChat
- LLaVA-based MLLM specialized for remote sensing imagery
- Supports visual grounding by predicting bbox coordinates as text tokens
- Base model used: `geochat-7B`

### 2.2 Bbox Tokenization (重要!)
GeoChat 的 bbox 是这种格式: `{<25><40><33><60>}`
关键事实: 这些数字是 **digit-by-digit** tokenized,每个数字字符 (`'2'`, `'5'`, `'4'`, `'0'` ...) 是一个独立的 token,**不是**一个整数 token。

### 2.3 Digit Token IDs (LLaMA tokenizer)
| Digit | Token ID |
|-------|----------|
| "0"   | 29900 |
| "1"   | 29896 |
| "2"   | 29906 |
| "3"   | 29941 |
| "4"   | 29946 |
| "5"   | 29945 |
| "6"   | 29953 |
| "7"   | 29955 |
| "8"   | 29947 |
| "9"   | 29929 |

These 10 IDs are the **only** token positions where soft labeling should be applied.

### 2.4 VRSBench
Remote sensing visual grounding benchmark. Visual grounding subset is used.

### 2.5 LoRA
Parameter-efficient fine-tuning. Used for both hard-label and soft-label experiments to keep the comparison fair.

---

## 3. Method

### 3.1 Hard Label Baseline (✅ Completed)
Standard cross-entropy fine-tuning of `geochat-7B` with LoRA on VRSBench. Serves as the control.

### 3.2 Soft Label Approach (✅ Implemented & working)
Replace one-hot target on digit positions with a **distance-weighted distribution** over the 10 digit tokens.

**Loss form (from `soft_label_loss.py`):**
```
L = (1 / (N_r + N_n)) · [ Σ_regular CE(hard) + λ · Σ_numerical CE(soft) ]
q^SL(t) = (1 − η)·δ(t) + η·ψ(t),  ψ ∈ {triangular, binomial, poisson, uniform}
```

**Main-experiment hyperparameters (used for the headline result):**
| Hyperparam | Value | Source |
|------------|-------|---------|
| Distribution shape ψ | **Triangular** | Wang et al. default / Table 4 |
| η (eta)              | **0.08**       | Wang et al. Fig. 2(c) optimum |
| λ (lambda)           | **2.0**        | Wang et al. Fig. 3 default (stable in [2, 5]) |

**Ablations (submitted 2026-04-22, running):**
ψ ∈ {triangular, binomial, poisson, uniform}, with η ∈ {0.02, 0.05, 0.10} variants. See `experiments/ablations/`.

### 3.3 Implementation Files

| File | Role |
|------|------|
| `geochat/train/soft_label_loss.py` | Build ψ distributions; compute combined CE loss with numeric/regular split |
| `geochat/model/language_model/geochat_llama.py` | Modified `forward()` — branches on `soft_label_enable` flag |
| `geochat/train/train.py` | Adds `soft_label_enable / _distribution / _eta / _lambda` training args and plumbing |
| `geochat/train/train_multinode.py` | Multi-node launcher variant (Isambard 4× GH200) |
| `experiments/train_softlabel_vrs.sh` | Main training SLURM script (Isambard) |
| `experiments/eval_softlabel_vg_isambard.sh` | Evaluation launcher |
| `experiments/ablations/*.sh` | Per-ψ / per-η ablation scripts |

The `soft_label_enable` flag is the single switch that conditionally activates the custom soft-label loss in the model's `forward()` pass.

**Key historical bugfix (2026-04-14)**: PeftModel attribute delegation — digit token IDs / soft-label matrix attached to the outer `PeftModel` wrapper were not reachable from the wrapped base-model's `forward()`. Fix: attach them to `model.base_model.model` (the true inner reference) and sync device in `forward()`. Recorded in commit `b1d9750`.

---

## 4. Current Results

Full results table: `experiments/vrsbench_eval/RESULTS_COMPARISON.md` (16,159 VRSBench samples, eval on 4× GH200).

### 4.1 Hard Label Baseline (matches official GeoChat numbers ✅)
| Metric        | Score   |
|---------------|---------|
| Acc@IoU 0.5   | 48.90%  |
| Acc@IoU 0.7   | 18.81%  |
| meanIoU       | 44.17%  |
| cumIoU        | 48.36%  |

This closely matches GeoChat's official benchmark — confirms experimental setup is correct.

### 4.2 Soft Label (✅ works; modest but consistent gains — 2026-04-14)

Headline (Triangular, η=0.08, λ=2.0, 5 epochs VRSBench):

| Subset | N | Acc@0.5 (H→S) | Acc@0.7 | meanIoU | cumIoU |
|---|---|---|---|---|---|
| **All** | 16,159 | 48.90 → **49.36** (+0.46) | +0.45 | +0.23 | −2.94 |
| **Unique** | 6,736 | 57.10 → **57.14** (+0.04) | +0.21 | +0.17 | −5.13 |
| **Non-unique** | 9,423 | 43.04 → **43.80** (+0.76) | **+0.64** | **+0.27** | **+0.90** |

**Finding**: Non-unique subset gains most on *all four* metrics — matches Wang et al. theory (distance-aware penalty helps when scene has multiple plausible targets). cumIoU regresses on Unique/All because a few very-large-bbox predictions inflate the global union denominator (per-sample metrics all improve).

**Context vs the paper**: We are in the paper's "specialist" regime (base model already fine-tuned on in-domain grounding data). Paper's LLaVA-13B-specialist gains on RefCOCOg val are +0.7 pp — our +0.46 pp sits in the same saturation band.

### 4.3 Ablations (in progress, 2026-04-22 submission)

Four SLURM jobs submitted: `triangular_eta005`, `uniform_eta005`, `poisson_eta002`, `binomial_eta005`. Check wandb `output.log` (NOT the SLURM `.out` file — wandb captures stdout) at `/projects/b6ar/dw22963/GeoChat/wandb/run-<id>/files/output.log` for loss history. Plus a `triangular_eta010` variant.

---

## 5. Known Issues / Watch-outs

### 5.1 Resolved: Soft Label Convergence Bug (fixed 2026-04-14, commit `b1d9750`)

Previously soft-label loss stayed ~0.75 and didn't converge. Root cause: PeftModel attribute delegation (attributes attached to the outer `PeftModel` wrapper were not reachable from the wrapped inner base-model's `forward()`). Fix: attach `digit_token_ids` and `soft_label_matrix` to `model.base_model.model` rather than to the outer `PeftModel`, and ensure device sync inside `forward()`. Broken predictions file from the old run is `predictions/vg_soft_grounding_predictions.jsonl` — do NOT compare against it.

### 5.2 Active: Torch LRScheduler monkey-patch caution

torch 2.11 added `strict=True` to `zip()` inside `LRScheduler._update_lr`, incompatible with transformers 4.37 + DeepSpeed ZeRO (param_groups are consolidated after scheduler creation). A minimal patch that forgets `self.last_epoch += 1` silently freezes the LR at `last_epoch = -1`, which for cosine-with-warmup produces a small **negative** LR — gradient *ascent*, loss blows up. Lost ~4 GPU-hours to this once. If patching torch internals in future torch upgrades, **replicate the full method body**, change only the offending line, and always sanity-check `'learning_rate':` in wandb output.log on the first steps.

### 5.3 Active: wandb captures stdout from 0.26+

With wandb enabled, training loss / LR lines do **not** appear in SLURM `.out` — they are redirected to `wandb/run-<ts>-<id>/files/output.log`. An empty `.out` is NOT a failure signal.

---

## 6. Key Insights / Important Findings

### 6.1 Fundamental Limitation of Per-Digit Soft Labeling 🔑
Distance signal is **per-digit**, not **per-value**.

**Example:** Coordinates "32" and "21"
- True numerical distance: |32 − 21| = 11
- Per-digit distances: |3−2| + |2−1| = 1 + 1 = 2
- Per-digit perceives this as "very close" when it's actually quite far

**Implication:** The loss cannot capture true multi-digit numerical distance. This is a **fundamental limitation worth discussing prominently in the dissertation Discussion chapter** — it explains why even a correctly-implemented soft label might not help much, and points to "predict whole-number tokens" or "predict per-coordinate" as future work.

### 6.2 Honest Framing of Negative Results
Project framing (supervisor approved): *"Honest diagnostic findings."*
- The negative result is itself a contribution — it identifies a previously-unstated limitation of the Wang et al. method when ported to coordinate prediction
- Do NOT oversell, do NOT bury the negative result

### 6.3 Baseline Match Validates Setup
Hard label baseline matching official GeoChat numbers confirms the entire experimental pipeline (data, eval, training) is correct. Any soft label issues are localized to the soft-label-specific code, not the broader system.

---

## 7. Dissertation Status

**Format:** Bristol CS dissertation template (cloned at `/projects/b6ar/dw22963/references/bristol-diss-template/`), 25–50 pages, standard academic chapter structure.

**Chapter progress:**
| Chapter | Status |
|---------|--------|
| 1. Introduction | ✅ outline drafted |
| 2. Background & Related Work | ✅ outline drafted |
| 3. Methods | ⏳ TODO |
| 4. Experiments | ⏳ TODO — now has positive headline result (not diagnostic-only); still highlight §6.1 per-digit limitation |
| 5. Discussion | ⏳ TODO (must include §6.1 limitation and §4.2 Non-unique subset finding) |
| 6. Conclusion | ⏳ TODO |

**Poster:** ✅ Completed. Used "honest diagnostic findings" framing. Supervisor feedback on layout, figure quality, loss curve consolidation, and soft label distribution visualization has been incorporated.

**Framing update**: The soft-label bug was fixed after the poster. Final report should present: (a) positive headline result on all / Non-unique subsets, (b) honest cumIoU regression, (c) §6.1 per-digit limitation as a discussion point (not a negative-result framing).

**Length guidance:**
- CS dissertation introductions are typically 1–2 pages — do NOT inflate
- Each chapter follows standard academic norms, not "maximum content"

---

## 8. Compute Environment

### 8.1 Current: Isambard AIP2 ⚙️

| Field | Value |
|---|---|
| Login host | (one of the `loginNN` nodes — currently on `login41`) |
| Username | `hanzzz.b6ar` (HPC account) / `dw22963` (Bristol project code) |
| Project allocation | `b6ar` (group), quota lives under `/lus/lfs1aip2/projects/b6ar/` |
| Work directory | `/projects/b6ar/dw22963/GeoChat` (symlink → `/lus/lfs1aip2/projects/b6ar/dw22963/GeoChat`) |
| Home directory | `/home/b6ar/hanzzz.b6ar` |
| Scheduler | SLURM (`sbatch` at `/usr/bin/sbatch`) |
| GPU type | **NVIDIA GH200** (Grace-Hopper, 4 per node) |
| conda / mamba | miniforge3 at `/projects/b6ar/dw22963/miniforge3/`, env name `geochat` |
| wandb run logs | `/projects/b6ar/dw22963/GeoChat/wandb/run-<ts>-<id>/files/output.log` |
| Large data / models | `/projects/b6ar/dw22963/data/` (24 GB), `/projects/b6ar/dw22963/models/` (42 GB) |
| Reference repos | `/projects/b6ar/dw22963/references/` (VRSBench, GeoChat-upstream, bristol-diss-template) |

### 8.2 Legacy: BluePebble (archived, no longer used)
- Login: `bp1-login.acrc.bris.ac.uk`
- User: `dw22963`
- Work dir: `/user/home/dw22963/work/GeoChat/`
- SSH key: `id_ed25519_bluepebble` (passphrase-free)
- Scheduler: SLURM, A100 SXM4 40GB GPUs (used for the main 5-epoch training run ~42 h)

---

## 9. Current Focus / 当前重点 🎯

**Phase:** Final dissertation writing (soft label already works — no longer a debugging phase)

**Primary task (in priority order):**
1. Draft Chapter 3 (Methods) — describe the loss form in §3.2, digit-token masking, LoRA setup
2. Draft Chapter 4 (Experiments) — present headline result (§4.2) + ablation table once the 2026-04-22 jobs finish
3. Draft Chapter 5 (Discussion) — (a) Non-unique subset finding, (b) cumIoU regression analysis, (c) §6.1 per-digit limitation
4. Draft Chapter 6 (Conclusion)
5. Polish Chapter 1 & 2 from outlines to full prose

**Secondary tasks:**
- Check ablation job status (see §4.3) and fold results into Chapter 4
- Error analysis on Unique-subset IoU=0 cases that drive the cumIoU regression (useful for Discussion)
- Upload evaluation predictions to HuggingFace as a companion artifact

**Out of scope (do NOT suggest these now):**
- New training experiments beyond the ψ / η ablation grid already submitted
- Trying other base models or other datasets
- Refactoring the codebase
- Exploring entirely different approaches (whole-number tokens, regression heads, etc.) — these belong in "Future Work"

---

## 10. Reference Materials

- **Reference paper:** Wang et al., *Enhancing Numerical Prediction of MLLMs with Soft Labeling*, ICCV 2025
- **Base model:** `geochat-7B`
- **Dataset:** VRSBench (visual grounding subset)
- **Codebase:** GeoChat (LLaVA-derived)
- **My fork structure:** four modified files listed in §3.3

---

## 11. Writing Workflow & Style Preferences

### 11.1 Output Language
- **Default to Chinese** for explanations and prose
- Keep technical terms in English (e.g., "cross-entropy", "LoRA", "soft label", "bbox")
- For dissertation chapter content: **full Chinese prose** — I will self-rephrase, then translate via Google Translate. This reduces AI-detection rates and ensures my own voice.
- For structural planning / outlines: detailed bullet outlines are fine

### 11.2 Response Style
- 复杂问题: think step-by-step before answering
- For code questions: explain **mechanics** deeply, not just provide a solution
- For bugs: lay out hypotheses and how to test each, don't just "try this fix"
- Honest about uncertainty — no rationalization, no fabrication

### 11.3 Citation Discipline (for dissertation)
- Every non-trivial claim needs a citation or be marked as "my contribution"
- The Wang et al. paper is the central citation — cite specific sections, not the whole paper

---

## 12. Glossary (quick reference)

| Term | Meaning in this project |
|------|-------------------------|
| **Hard label** | Standard one-hot cross-entropy target |
| **Soft label** | Distance-weighted distribution over digit tokens |
| **Digit token** | A single character `'0'`–`'9'` as a separate LLaMA token |
| **Bbox token** | Sequence of digit tokens forming `{<x1><y1><x2><y2>}` |
| **Visual grounding (VG)** | Predict bbox given image + textual description |
| **LoRA** | Low-Rank Adaptation, parameter-efficient fine-tuning |
| **PEFT** | Parameter-Efficient Fine-Tuning library (huggingface) |
| **VRSBench** | Visual remote sensing benchmark |
| **GeoChat-7B** | LLaVA-based MLLM for remote sensing, 7B parameter base |
| **η (eta)** | Soft label distribution smoothness param |
| **λ (lambda)** | Soft label loss weight / temperature param |
| **Acc@IoU 0.5 / 0.7** | Visual grounding accuracy metric |

---

## 13. Update Log

- **2026-04-23:** Initial handover document created. Project migrated from BluePebble to Isambard. Section 8.1 needs filling.
- **2026-04-23 (later):** Moved CLAUDE.md from `docs/report_context/` to repo root so Claude Code auto-loads it every session. Updated §3.2/§3.3 (soft label now working; λ=2.0 not 4.0; correct file paths), §4 (replaced "NOT working" with actual headline + subset table), §5 (rewrote as "known issues" — bug resolved, added torch scheduler + wandb watch-outs), §7 (framing now "positive result with honest caveats"), §8.1 (filled in Isambard fields), §9 (refocused on writing, removed debug priority). Reference repos cloned to `/projects/b6ar/dw22963/references/` (VRSBench, GeoChat-upstream, bristol-diss-template).
