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

## Results on EG Subset (1118 samples)

| System                   | Type                     | BLEU | chrF | chrF++ | Semantic Similarity |
| -------------------------| -------------------------| ---: | ---: | -----: | ------------------: |
| Gemma-4-E2B-it Base      | Baseline / no fine-tuning| **14.55** | **44.96** | **41.91** | **0.952464** |
| Gemma-4-E2B-it FNN-r8 MLP| LoRA fine-tuned          |13.37 | 43.16 | 40.14 | 0.949918 |
| Qwen3.5-2B LoRA all-r16  | LoRA fine-tuned          |8.56 | 37.91 | 34.57 | 0.943186 |
| Qwen3.5-2B Base          | Baseline / no fine-tuning|2.37 | 26.52 | 23.24 | 0.923324 |

### Evaluation Metrics

### BLEU

BLEU measures how much the generated translation overlaps with the reference translation at the word n-gram level. It mainly rewards exact word/phrase matches and applies a brevity penalty to avoid overly short translations.

Higher BLEU is better.

```text
BLEU = BP × exp( Σ_{n=1}^{N} w_n log p_n )
````

where:

```text
p_n = modified precision for n-grams of order n
w_n = weight for each n-gram order, usually 1/N
BP  = brevity penalty
```

The brevity penalty is:

```text
BP = 1              if c > r
BP = exp(1 - r/c)  if c <= r
```

where:

```text
c = generated translation length
r = reference translation length
```

---

### chrF

chrF is a character n-gram F-score between the generated translation and the reference. It is useful for Arabic and other morphologically rich languages because it can reward partial word overlap even when exact word matching fails.

Higher chrF is better.


                            $chrF_β = (1 + β²) × (chrP × chrR) / (β² × chrP + chrR)$


where:

[\
chrP = character n-gram precision
chrR = character n-gram recall
β    = recall weight, commonly β = 2
\]

---

### chrF++

chrF++ extends chrF by combining character n-gram matching with word n-gram matching. It keeps the flexibility of character-level evaluation while adding sensitivity to word-level correctness.

Higher chrF++ is better.

```text
chrF++ = F-score over character n-grams and word n-grams
```

Conceptually:

```text
chrF++ = F_β(char_ngram_precision/recall + word_ngram_precision/recall)
```

---

### Semantic Similarity

Semantic similarity measures how close the generated translation and the reference are in embedding space. Unlike BLEU and chrF, it can capture meaning similarity even when the wording is different.

In these experiments, semantic similarity is computed using E5-large embeddings and cosine similarity.

Higher semantic similarity is better.

```text
similarity = cos(e_pred, e_ref)
```

where:

```text
e_pred = embedding of the generated translation
e_ref  = embedding of the reference translation
```

Cosine similarity is computed as:

```text
cos(e_pred, e_ref) = (e_pred · e_ref) / (||e_pred|| ||e_ref||)
```



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



