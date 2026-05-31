# LahjaMT: English-to-Arabic Dialect Machine Translation

This repository contains experiments and comparisons of baselines behavior and LoRA adaptation for  **English → Arabic dialect machine translation**, with the broader goal of building and evaluating context-aware translation systems for dialectal Arabic.

The current experiments focus on the **Egyptian Arabic (EG) subset**, but the intended direction is broader English-to-dialectal-Arabic MT.

## Task

Given an English dialogue turn, optionally with previous dialogue context and metadata, the model generates a natural Arabic dialect translation.

The current prompt style is:

```text
Translate into natural Egyptian Arabic dialect.
Do not use Modern Standard Arabic unless unavoidable.
Return only the Arabic translation.
````

For now, the evaluated dialect is **Egyptian Arabic**, but the same framework is to be extended to the rest of dialects.

## Current Evaluation Setup

The current reported results are on the **English → Egyptian Arabic EG subset**.

```text
Eval examples: 1118
Metrics: BLEU, chrF, chrF++
```

Best systems are compared against each other:

* Gemma-4-E2B-it baseline without fine-tuning
* Gemma-4-E2B-it LoRA fine-tuned on MLP/FNN modules with rank 8
* Qwen3.5-2B LoRA fine-tuned with all-module LoRA rank 16
* Qwen3.5-2B baseline without fine-tuning

- More experiments are in the notebooks

## Results on EG Subset

| System                    | Type                      | Examples |      BLEU |      chrF |    chrF++ |
| ------------------------- | ------------------------- | -------: | --------: | --------: | --------: |
| Gemma-4-E2B-it Base       | Baseline / no fine-tuning |     1118 | **14.55** | **44.96** | **41.91** |
| Gemma-4-E2B-it FNN-r8 MLP | LoRA fine-tuned           |     1118 |     13.37 |     43.16 |     40.14 |
| Qwen3.5-2B LoRA all-r16   | LoRA fine-tuned           |     1118 |      8.56 |     37.91 |     34.57 |
| Qwen3.5-2B Base           | Baseline / no fine-tuning |     1118 |      2.37 |     26.52 |     23.24 |

## Main Observations

The strongest current system on the EG subset is the **Gemma-4-E2B-it baseline**, without fine-tuning:

```text
BLEU   = 14.55
chrF   = 44.96
chrF++ = 41.91
```

Fine-tuning had different effects depending on the model.

For **Qwen3.5-2B**, LoRA produced a large improvement:

```text
Qwen base BLEU: 2.37
Qwen LoRA BLEU: 8.56
```

For **Gemma-4-E2B-it**, the baseline was already strong, and MLP/FNN-r8 LoRA slightly degraded performance:

```text
Gemma base BLEU: 14.55
Gemma FNN-r8 BLEU: 13.37
```

## Interpretation

The current results suggest that model adaptation behaves differently depending on the base model suggesting that data size may be a critical player in the finetuning process.

**Qwen3.5-2B** starts from a weak baseline, so LoRA fine-tuning gives a clear improvement.

**Gemma-4-E2B-it** already performs strongly as a baseline, so small-data fine-tuning may cause slight degradation rather than improvement.

This supports the conclusion that fine-tuning is not automatically beneficial for every base model, especially when the base model already has strong translation and instruction-following abilities.



