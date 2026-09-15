# LahjaMT: Context-Aware English-to-Dialectal-Arabic Machine Translation

**LahjaMT** is a compact system for translating English dialogue into **thirteen country-level Arabic
varieties**. It adapts the 3.09B-parameter [**NileChat-3B-Base**](https://huggingface.co/UBC-NLP/NileChat-3B-Base)
with LoRA, conditions translation on dialogue history and metadata, and routes each dialect to a
checkpoint–prompt *expert* selected on held-out data.

This is the official code for our system description paper at the **AlexandriaX-2026 Shared Task on
Dialectal Arabic Machine Translation** (Subtask 1), presented at the Fourth Arabic NLP Conference.

> LahjaMT ranked **2nd in the constrained track** (28.30 spBLEU) and **3rd in the unconstrained track**
> (28.54 spBLEU), outperforming our own inference-only configurations built on the far larger
> `gpt-oss-20b` and `gpt-oss-120b` models — underscoring the value of task-specific adaptation over scale.

- 📄 **Paper:** `LahjaMT at AlexandriaX-2026` (camera-ready included in the release notes)
- 🤗 **Model (LoRA adapters + routing + inference):** [`MohamedAbdallah98/LahjaMT`](https://huggingface.co/MohamedAbdallah98/LahjaMT)
- 💻 **Code:** this repository

---

## Official results (private test, macro-averaged over 13 dialects)

| Track          | Rank | spBLEU | chrF++ |
| -------------- | :--: | -----: | -----: |
| Constrained    |  2   |  28.30 |  43.81 |
| Unconstrained  |  3   |  28.54 |  44.02 |

The unconstrained system changes **only** the Libyan (LY) and Sudanese (SD) routes, replacing them with
specialists trained on auxiliary [SMOL](https://aclanthology.org/2025.wmt-1.94/) data; the other eleven
routes are identical to the constrained submission.

---

## How it works

LahjaMT is built offline as an **expert bank** and applied at inference by deterministic per-dialect routing.

### 1. Backbone and adaptation
- Base model: **NileChat-3B-Base** (a decoder-only model continuing the pretraining of Qwen2.5-3B, centered on Egyptian and Moroccan Arabic).
- Adapter: **LoRA** with `r = 16`, `α = 32`, dropout `0.05`, applied to **all seven projection modules** in every Transformer block (attention + feed-forward).
- Cost: ~29.9M trainable parameters (**0.96%** of the 3.12B-parameter model), a **~114 MB** adapter — cheap enough to keep many checkpoints as experts.

### 2. Context-aware input
Each instance is a single instruction block containing: two English–Arabic demonstrations, metadata
(country, dialect, domain, speaker, gender direction, optional persona/roles), up to three preceding
English turns, and the current English turn, followed by seven explicit output rules. Loss is applied
only to the target response tokens. The full template is in Appendix A of the paper.

Three prompt configurations define, together with a LoRA checkpoint, an **expert**:
- **P1** — deterministic same-country/same-domain demonstrations (matches the fine-tuning format).
- **P2** — semantically retrieved demonstrations (all-MiniLM-L6-v2 over the English source).
- **P3** — retrieval **plus** persona / participant-role metadata.

### 3. Two-stage training
- **Stage 1:** fine-tune on 66,480 training turns — 3 epochs, LR `2e-4`, cosine schedule, warm-up `0.03`, weight decay `0.01`, effective batch 8, max length 2,048, BF16. The best development checkpoint becomes the parent adapter.
- **Stage 2:** continue from the Stage-1 parent on 20,920 turns (12,250 dev + 8,670 continuation from the public-test split) — 3 epochs, LR `2e-5`, other settings unchanged. **Six** checkpoints are retained.

All runs use BF16 on a single **NVIDIA RTX 5090**.

### 4. Expert bank and dialect routing
The six Stage-2 checkpoints × three prompt configurations form an **18-candidate expert bank**. A routing
table is built on the 5,772-turn held-out selection set with a conservative rule: a dialect is moved off
the default checkpoint only when its best checkpoint–prompt pair beats the default by **≥ 0.15 spBLEU**.
Seven dialects keep the default; six (OM, YE, LY, MR, SA, TN) are promoted.

### 5. Inference
The provided dialect label deterministically retrieves the checkpoint–prompt pair. Generation everywhere:
beam search with **4 beams**, length penalty `1.0`, repetition penalty `1.05`, up to `120` new tokens,
no sampling. No Arabic normalization is applied; post-processing only strips template markers and
predominantly Latin-script lines.

---

## Final routing map (held-out spBLEU / chrF++)

Adapters are Stage-2 continuation checkpoints (by optimizer step). The two `+ SMOL spec.` rows apply
only to the unconstrained submission.

| Variety | Route            | spBLEU | chrF++ |
| :------ | :--------------- | -----: | -----: |
| EG      | 5,200 + P3       |  31.88 |  45.60 |
| JO      | 5,200 + P1       |  35.50 |  49.12 |
| LB      | 5,200 + P1       |  32.29 |  46.00 |
| LY      | 6,400 + P3       |  23.39 |  39.01 |
| LY *(unconstr.)* | + SMOL spec. | 23.86 | 39.14 |
| MA      | 5,200 + P2       |  23.31 |  39.77 |
| MR      | 2,000 + P1       |  17.94 |  34.34 |
| OM      | 4,900 + P1       |  32.57 |  47.11 |
| PS      | 5,200 + P1       |  34.26 |  48.26 |
| SA      | 1,600 + P2       |  35.25 |  49.78 |
| SD      | 5,200 + P1       |  26.15 |  40.97 |
| SD *(unconstr.)* | + SMOL spec. | 27.43 | 42.06 |
| SY      | 5,200 + P3       |  39.38 |  53.19 |
| TN      | 7,600 + P1       |  35.23 |  47.61 |
| YE      | 4,900 + P1       |  25.23 |  41.71 |
| **Macro (constrained)**   |  | **30.18** | **44.81** |
| **Macro (unconstrained)** |  | **30.32** | **44.90** |

---

## Repository structure

The project is organized as a set of Jupyter notebooks documenting each stage of the work.

| Folder | Contents |
| :----- | :------- |
| `EG_Ablation_Experiments/` | Early Egyptian-Arabic ablations: LoRA module/rank sweeps, backbone comparisons (NileChat / Qwen / Gemma), NTK experiments, corrector/router variants, and metric comparisons. |
| `All_Dialects_Experiments/` | Scaling the approach to all 13 dialects: NileChat LoRA training and few-shot beam-4 inference. |
| `MERGE_DEV/` | Stage-2 continuation training on the merged development + public-test data. |
| `Inference_Variants/` | Routing and inference studies: expert-bank diagnostics, cross-encoder candidate reranking, specialist refinement, submission generation, and the negative-result variants (MBR/MoE, out-of-fold router, reference-based oracle). |
| `Server_NoteBooks/` | Server-side (Nawy) runs, including the `gpt-oss-20b` inference-only baseline and continuation training. |
| `assets/learning_curves/` | Training-loss, eval-loss, and learning-rate curves. |

> **Note:** the notebooks were run across Colab and the Nawy server, so some paths and installs are
> environment-specific. They are provided for transparency and reproducibility of the reported results.

---

## Using the model

The trained LoRA adapters, the routing table, prompt templates, and an inference script are published on
Hugging Face: [`MohamedAbdallah98/LahjaMT`](https://huggingface.co/MohamedAbdallah98/LahjaMT). See the model card
there for load-and-generate instructions (PEFT over `UBC-NLP/NileChat-3B-Base`).

The published model routes **each dialect to its best expert** — the full-power configuration, with the
Libyan/Sudanese SMOL specialists included. Users select only a target dialect; the constrained vs.
unconstrained *tracks* below are a shared-task distinction, not something you choose at inference.

---

## Citation

If you use LahjaMT, please cite:

```bibtex
@inproceedings{abdallah2026lahjamt,
  title     = {{LahjaMT} at {AlexandriaX-2026}: A Context-Aware English-to-Dialectal Arabic Machine Translation System with Lightweight Routing of {LoRA} Experts},
  author    = {Abdallah, Mohamed A. and El-Beltagy, Samhaa R.},
  booktitle = {Proceedings of the Fourth Arabic Natural Language Processing Conference: Shared Tasks},
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics}
}
```

## Acknowledgments

We gratefully acknowledge the **Nawy AI lab** for providing the computational resources that supported
this work.
