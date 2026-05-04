# AerialGPT: Soft Labeling on VRSBench Visual Grounding — Results Comparison

**Date**: 2026-04-14
**Model**: GeoChat-7B (Vicuna-1.5 + CLIP ViT-L/14 + LoRA)
**Dataset**: VRSBench-EVAL referring split (16,159 samples, 2,020 unique images)
**Training**: 5 epochs on VRSBench, 4× A100 SXM4 40GB, ~42 hours
**Evaluation**: 4× NVIDIA GH200 (Isambard AIP2), ~17 minutes for 16,159 samples

---

## 1. Headline Results

| Method | Acc@IoU=0.5 | Acc@IoU=0.7 | meanIoU | cumIoU |
|---|---|---|---|---|
| **Hard Label (Baseline)** — standard CE loss | 48.90% | 18.81% | 44.17% | **48.36%** |
| **Soft Label (Ours)** — η=0.08, λ=2, Triangular | **49.36%** | **19.26%** | **44.40%** | 45.42% |
| **Δ (Soft − Hard)** | **+0.46** | **+0.45** | **+0.23** | −2.94 |

Soft labeling improves accuracy and meanIoU on the full 16,159-sample evaluation. cumIoU regresses; analysis in §4.

---

## 2. Breakdown by Uniqueness Subset

VRSBench annotates each sample as `unique=True` (only one object in the image matches the description) or `unique=False` (multiple candidates, model must disambiguate). Soft labeling should in principle help most where numerical ambiguity exists.

### 2.1 Unique subset (6,736 samples)

| Metric | Hard | Soft | Δ |
|---|---|---|---|
| Acc@0.5 | 57.10% | 57.14% | +0.04 |
| Acc@0.7 | 21.36% | 21.57% | +0.21 |
| meanIoU | 49.81% | 49.98% | +0.17 |
| cumIoU | 53.50% | 48.37% | −5.13 |

### 2.2 Non-unique subset (9,423 samples)

| Metric | Hard | Soft | **Δ** |
|---|---|---|---|
| Acc@0.5 | 43.04% | **43.80%** | **+0.76** |
| Acc@0.7 | 16.98% | **17.62%** | **+0.64** |
| meanIoU | 40.14% | **40.41%** | **+0.27** |
| cumIoU | 37.35% | **38.25%** | **+0.90** |

**Key finding**: Non-unique subset benefits *most*, across *all four metrics* (including cumIoU). This is the theoretically-motivated prediction of soft labeling — when the scene contains multiple plausible targets, finer-grained numerical supervision helps the model pick the correct one.

---

## 3. Context — How Does This Compare to the Original Paper?

Wang et al. ICCV 2025 reports soft-label gains on **RefCOCO/+/g** (natural images), using **Acc@0.5 only**:

| Base Model | Setting | Hard | Soft | **Δ (Acc@0.5)** |
|---|---|---|---|---|
| LLaVA-7B | generalist (RefCOCOg val) | 72.1 | 78.5 | **+6.4** |
| LLaVA-7B | specialist (RefCOCOg val) | 79.0 | 82.0 | **+3.0** |
| LLaVA-13B | generalist (RefCOCOg val) | 78.0 | 79.2 | +1.2 |
| LLaVA-13B | **specialist** (RefCOCOg val) | 86.0 | 86.7 | **+0.7** |

**Why our gains are smaller:**

GeoChat is **not** a fresh LLaVA — it is a model that has already undergone full visual-grounding fine-tuning on remote-sensing data. It corresponds to the paper's "**specialist**" setting — a model whose grounding capability has already been strongly optimized on in-domain data. In this regime, the paper itself reports **+0.7 pp on LLaVA-13B specialist**, and our **+0.46 pp on GeoChat-7B** sits in the same saturation band.

The paper's large gains (+3 to +6 pp) come from the **generalist** setting — starting from a general-purpose LLaVA and learning grounding from weaker initialization. The soft-label method has the most *room* to help when the base is weakest.

**Interpretation**: Our experiment confirms soft labeling generalizes to the remote-sensing domain, but like all regularization-style methods, diminishing returns on already-strong specialists.

---

## 4. Why cumIoU Regresses on Unique / All (but Improves on Non-unique)

**cumIoU = Σᵢ Intersectionᵢ / Σᵢ Unionᵢ** — computed globally across all samples. It is **not** the per-sample IoU averaged; it is the *dataset-level* ratio. A few predictions with very large bounding boxes (huge union) can dominate the denominator.

Observations:
- Acc@0.5 / Acc@0.7 / meanIoU — all **per-sample** metrics. They improve uniformly.
- cumIoU — **aggregate** metric. Regresses on Unique/All, improves on Non-unique.

This asymmetry suggests: in a small fraction of Unique samples, the soft-label model produces bounding boxes that are *too large* (high recall, low precision in terms of area), inflating the global union sum without comparable gains in intersection. Investigating these outlier cases is a natural next step (error analysis by bbox size / object class).

---

## 5. Why Non-unique Benefits Most (Theoretical Alignment)

Wang et al. frame soft labeling as *implicitly introducing distance-aware penalties* into classification loss. Prediction errors close to the target (e.g., predicting `247` when truth is `248`) are penalized less than distant errors (e.g., predicting `100`).

In VRSBench:
- **Unique samples**: one object matches → hard label is already near-perfect supervision; soft label adds little.
- **Non-unique samples**: model must choose among multiple candidates → coordinate predictions are more likely to be *close but wrong* → distance-aware loss provides the strongest gradient signal precisely here.

Non-unique gains of **+0.76 Acc@0.5** and **+0.64 Acc@0.7** are the empirical signature predicted by the paper's theory.

---

## 6. Hyperparameter Setup

| Hyperparameter | Value | Rationale |
|---|---|---|
| Soft distribution ψ | **Triangular** | Paper Table 4: ties Binomial for best; Triangular is paper default |
| η (mixture ratio) | **0.08** | Paper optimum on RefCOCOg val, Fig. 2(c) |
| λ (numeric-vs-regular token loss balance) | **2.0** | Paper Fig. 3: stable in [2, 5], default is 2 |
| Soft label support | digit tokens "0"–"9" only | Following the paper's scope |
| LoRA rank r | 64 | Inherited from GeoChat official training |
| Base LLM | Vicuna-1.5-7B (via GeoChat-7B checkpoint) | — |
| Vision tower | CLIP ViT-L/14-336 (interpolated to 1297 positions) | GeoChat convention |

---

## 7. Key Training / Evaluation Numbers

- **Training loss**: 3.77 (init) → 0.73 (final epoch 5). Average train_loss across run = 0.98. **Converged normally.**
- **Training hardware**: 4× A100 SXM4 40GB (BluePebble HPC)
- **Training time**: ~42 hours for 5 epochs on full VRSBench train split
- **Evaluation hardware**: 4× NVIDIA GH200 (Isambard AIP2)
- **Evaluation time**: ~17 minutes for 16,159 test samples (4-way parallel, ~1.03 it/s per GPU)
- **LoRA-merged model size**: ~14 GB (uploaded to HuggingFace `Hanzzz11/geochat-vrs-softlabel-merged`)

---

## 8. Summary for Poster

✅ **Research question answered**: Soft labeling *does* improve visual grounding on remote-sensing data — gains are modest on an already-strong specialist model, but **theoretically-motivated: the non-unique subset where soft labeling should help most, does.**

✅ **Methodological validation**: All per-sample metrics (Acc@0.5, Acc@0.7, meanIoU) improve consistently across splits.

⚠️ **Nuance to discuss**: cumIoU regresses on Unique subset — requires error analysis (future work).

---

## References

1. Wang, Cai, Yang, Modolo, Swaminathan. *Enhancing Numerical Prediction of MLLMs with Soft Labeling.* ICCV 2025.
2. Kuckreja et al. *GeoChat: Grounded Large Vision-Language Model for Remote Sensing.* CVPR 2024.
3. Li et al. *VRSBench: A Versatile Vision-Language Benchmark for Remote Sensing Image Understanding.* NeurIPS 2024.
