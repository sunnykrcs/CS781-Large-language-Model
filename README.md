# In-House Legal Intelligence System: a Legal Language Model Built from Scratch

**CS781: Advanced NLP, Large Language Models, IIT Kanpur (Capstone Project)**


A law firm that cannot send privileged case files to external APIs needs a **self-hosted** legal language system. This repository contains a seven-phase pipeline for Indian legal text. It starts from a custom tokenizer and hand-written transformer architectures pre-trained from scratch (no pre-trained weights, no external legal corpora). It then adapts each backbone to six downstream legal NLP tasks using full fine-tuning and parameter-efficient methods (LoRA, LoRMA, adapters, prefix/soft prompts, BitFit, QLoRA, distillation).

> **Author:** Sunny Kumar Pandit, Roll No. 231055, IIT Kanpur ([sunnyp23@iitk.ac.in](mailto:sunnyp23@iitk.ac.in))
> **Full write-up:** [`Report/CS781-CP-Sunny-Pandit-231055-report.pdf`](Report/CS781-CP-Sunny-Pandit-231055-report.pdf)
> **Dataset:** [Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone) (Hugging Face)

---

## Table of Contents

1. [Highlights](#highlights)
2. [Competition results](#competition-results)
3. [Repository structure](#repository-structure)
4. [The pipeline at a glance](#the-pipeline-at-a-glance)
5. [Phase 1: Foundation models (pre-training)](#phase-1-foundation-models-pre-training)
6. [Phases 2 to 7: Downstream tasks](#phases-2-to-7-downstream-tasks)
7. [Adaptation methods](#adaptation-methods)
8. [Results in detail](#results-in-detail)
9. [Key findings and lessons](#key-findings-and-lessons)
10. [Getting started](#getting-started)
11. [Notebook anatomy and configuration knobs](#notebook-anatomy-and-configuration-knobs)
12. [Known limitations](#known-limitations)
13. [Future work](#future-work)
14. [LLM usage declaration](#llm-usage-declaration)
15. [References](#references)

---

## Highlights

- **Four hand-written architecture families**, each as a *baseline* and a *modified* variant, all in the 50M to 300M parameter budget (parameter counts are asserted in code):
  decoder-only (LLaMA-style), local/global hybrid, gated linear-attention hybrid, and encoder-decoder.
- **Custom 48k byte-level BPE tokenizer** with a word-end marker, shared across every phase.
- **Pre-trained from scratch on 400M tokens** of Indian legal text (Supreme Court, High Courts, Tribunals and Acts), with resumable checkpoints so training survives Kaggle's 12-hour session cap.
- **Seven tasks** end to end: language modelling, NER, rhetorical roles, judgment prediction, statute identification, summarization and prior case retrieval.
- **All adaptation methods are hand-written** for plain `nn.Linear` layers, each with a sanity test that the adapted model starts exactly at the Phase-1 function.
- **Fast generation**: a KV cache plus a one-token-per-step recurrent state for the linear-attention model.
- **Architecture 1 (decoder-only) ranked best of our four systems in all seven phases.**

---

## Competition results

Best valid official score per phase (Codabench user `s_231055`). BS-F1 = BERTScore F1.

| Phase | Task | Best system | Primary metric (value) | Rank | Valid submissions |
|:---:|---|---|---|:---:|:---:|
| 1 | Legal LM pre-training | Arch 1, modified | BS-F1 (**0.8449**) | 8 | 23 |
| 2 | Legal NER | Arch 1 + LoRMA | strict macro-F1 (**0.5610**) | 39 | 23 |
| 3 | Rhetorical roles | Arch 1 + LoRA | macro-F1 (**0.5102**) | 51 | 18 |
| 4 | Judgment prediction | Arch 1 + LoRMA | macro-F1 (**0.7492**) | 49 | 7 |
| 5 | Statute identification | Arch 1 + LoRMA | macro-F1 (**0.3559**) | 86 | 8 |
| 6 | Summarization | Arch 1 + LoRA | BS-F1 (**0.8437**) | 50 | 4 |
| 7 | Prior case retrieval | Arch 1 + LoRA | micro-F1@7 (**0.2304**) | 119 | 4 |

87 valid submissions in total. Four architectures were run in every phase, so at least four distinct systems exist per phase.

---

## Repository structure

```
CS781-Large-language-Model/
├── README.md
├── Phase_1/                      # Pre-training + test-prompt generation
│   ├── Arch_1.ipynb              #   LLaMA-style decoder-only
│   ├── Arch_2.ipynb              #   Local/global (sliding-window) hybrid
│   ├── Arch_3.ipynb              #   Linear-attention / recurrent hybrid
│   └── Arch_4.ipynb              #   Encoder-decoder
├── Phase_2/  Arch_{1..4}.ipynb   # Legal NER (L-NER)
├── Phase_3/  Arch_{1..4}.ipynb   # Rhetorical Role prediction (RR)
├── Phase_4/  Arch_{1..4}.ipynb   # Court Judgment Prediction (CJPE)
├── Phase_5/  Arch_{1..4}.ipynb   # Legal Statute Identification (LSI)
├── Phase_6/  Arch_{1..4}.ipynb   # Abstractive Summarization
├── Phase_7/  Arch_{1..4}.ipynb   # Prior Case Retrieval (PCR)
└── Report/
    └── CS781-CP-Sunny-Pandit-231055-report.pdf
```

There are **28 notebooks**: 7 phases × 4 architectures. Every notebook is self-contained. The tokenizer, model code and PEFT modules are repeated inside each notebook, so no shared Python package is needed.

---

## The pipeline at a glance

```
                       ┌──────────────────────────────────────────────────────────┐
  48k byte-level BPE   │   Phase-1 backbone (one of Arch 1 to 4)                  │
  tokenizer  ────────► │   pre-trained from scratch on 400M legal tokens          │
                       └───────────────┬──────────────────────────────────────────┘
                                       │  weights copied into each task model
        adapt: full / LoRA / LoRMA / adapter / prefix / soft prompt / BitFit / QLoRA / KD
                                       │
   ┌───────────┬───────────┬───────────┼───────────┬────────────┬───────────┐
   P2 NER      P3 Roles    P4 CJPE     P5 LSI      P6 Summ.     P7 PCR
   BiGRU+BIO   BiGRU over  attention   per-statute LM head      cosine
   tags        sentences   over chunks window attn (frozen+PEFT) retrieval
```

Every downstream task descends **separately** from the Phase-1 checkpoint. There is no pipelining between phases. Each task gets a task head trained from scratch. Decoder backbones are causal, so heads that need right-hand context add a small bidirectional GRU (Phases 2, 3) or pool windows with attention (Phases 4, 5). For the encoder-decoder (Arch 4), only the encoder is used in Phases 3 to 5 and 7.

---

## Phase 1: Foundation models (pre-training)

**Task.** Train a 50M to 300M parameter language model from scratch on legal text. Given a 512-word prompt, the model writes a 128-word continuation, scored with **BERTScore F1** against the hidden reference on 1,000 test prompts. Validation perplexity is tracked as well.

### Data

| Source | Share of token budget (Arch 1, 3, 4) | Share (Arch 2) |
|---|:---:|:---:|
| Supreme Court | 40% | 50% |
| Acts | 30% | 25% |
| High Courts | 20% | 15% |
| Tribunals | 10% | 10% |

The raw High Court and Tribunal files (10.7 GB and 6.5 GB) dwarf the Supreme Court file (1.3 GB), so token budgets are drawn **by source** rather than in proportion to file size. A 5% random share of documents is held out for validation. Training uses 400M tokens, one epoch, with a 2,048-token context.

### Shared design

All models use pre-norm **RMSNorm**, **SwiGLU** feed-forward blocks (d_ff = 2048), d_model = 768 and 12 heads.

| # | Architecture | Baseline | Modified variant |
|:---:|---|---|---|
| **1** | LLaMA-style decoder | 12 layers, RoPE, multi-head attention, untied embeddings (158.7M) | + GQA (4 KV heads), QK-norm, tied embeddings, value residuals, 2 multi-token-prediction heads (weight 0.3), source embedding; savings re-spent on depth, giving **17 layers (145.0M)** |
| **2** | Local/global hybrid | Mistral-style sliding window of 512 in every layer (158.7M) | Window grows 128 to 512 with depth; every 4th layer is full attention with NoPE; GQA, QK-norm, tied embeddings (145.0M) |
| **3** | Linear attention | RetNet-style retention, fixed per-head decay (165.8M) | GLA-style: input-dependent decay gates, width-4 causal conv on q/k/v, L2-normalized q/k, output gate, softmax NoPE layer every 4th layer, 16 layers (154.3M); chunkwise-parallel training (chunk 64) plus a recurrent state for generation |
| **4** | Encoder-decoder | 6+6 layers, cross-attention in every decoder layer (172.8M) | 7 encoder + 8 decoder layers, cross-attention every 2nd decoder layer, 2× compressed encoder memory, GQA, QK-norm, tied embeddings, MTP (189.4M). The first 768 tokens of each 2,048-token block go to the bidirectional encoder and the decoder models the remaining 1,280 |

### Training recipe

- **Loss:** next-token cross-entropy (+ MTP terms for modified variants)
- **Optimizer:** AdamW, β = (0.9, 0.95), weight decay 0.1, gradient clipping 1.0
- **Schedule:** linear warm-up then cosine decay to 0.1× the peak LR
- **Peak LR:** 6×10⁻⁴ (Arch 1), 5×10⁻⁴ (Arch 2), 4×10⁻⁴ (Arch 3 and 4)
- **Precision and memory:** fp16 with loss scaling, gradient checkpointing
- **Hardware:** 2× Tesla T4 on Kaggle (Arch 1 run: 1,525 optimizer steps at 262,016 tokens per step)
- **Resumability:** a full checkpoint (model, optimizer, scheduler clock) is saved every 250 steps to `resume.pt`, so a run can continue across 12-hour sessions

### Decoding for the test set

| Arch | Strategy |
|---|---|
| 1, 2, 4 | **Minimum-Bayes-Risk (MBR)** selector: draws 16 samples (T = 0.4, top-k 20, top-p 0.88, repetition penalty 1.08) and keeps the sample with the highest average agreement with the others (0.9 × BERTScore + 0.1 × ROUGE-L) |
| 3 | Direct sampling (T = 0.1, top-k 50, top-p 0.95, no repetition penalty) |

Continuations are capped at 128 words and trimmed at a word boundary. Each notebook writes `prediction.json` and `submission.zip`.

---

## Phases 2 to 7: Downstream tasks

| Phase | Task | Metric | Data (train / dev / test) | Head and approach |
|:---:|---|---|---|---|
| **2** | **Legal NER**: 12 entity types as typed character spans | Strict span-level macro-F1 | 70 docs (8 held out locally) / – / 35 | Token classification with BIO tags (25 tags), BiGRU over backbone states, class-weighted CE, entity-free windows down-sampled; Viterbi decoding constrained to BIO rules plus a confidence threshold τ |
| **3** | **Rhetorical roles**: 13 roles per sentence (Corporate Law + Information Technology) | Macro-F1 | 16,845 / 2,142 / 2,197 sentences | Core-sentence windows (768 tokens + 256 left context), sentence vectors (mean + last token + domain embedding), 2-layer BiGRU over sentences, label smoothing 0.05 |
| **4** | **Court Judgment Prediction (CJPE)**: binary outcome from a full Supreme Court case | Macro-F1 | 37,387 / 1,988 / 1,517 cases | Overlapping 1,024-token chunks (stride 896, head and tail kept for very long cases), hierarchical head with relative-position embeddings and learned attention pooling, threshold tuned on dev, TF-IDF baseline as a sanity check |
| **5** | **Legal Statute Identification (LSI)**: multi-label over 100 statutes | Macro-F1 | 42,750 / 10,181 / 13,019 docs | 512-token windows (stride 384, at most 8), mean-pooled; per-statute label-wise attention over windows; BCE with clipped pos-weight; one global threshold tuned on dev |
| **6** | **Abstractive summarization** of long judgments | BERTScore F1 (+ ROUGE-1/L) | 6,679 / 351 / 100 docs | `document <sep> summary` continuation from the Phase-1 LM head (frozen backbone, 0.01 to 0.8% of parameters trained); input keeps first 60% and last 40% of 1,024 tokens; greedy decoding with repetition penalty 1.15 and no repeated 3-grams, up to 256 tokens |
| **7** | **Prior Case Retrieval (PCR)**: top-7 prior cases per query | Micro-F1@7 | 827 / 118 / 237 queries | Bi-encoder over up to four 512-token chunks, mean-pooled and averaged to a case vector, cosine similarity; InfoNCE with in-batch negatives plus one mined hard negative; each split uses its own candidate pool |

### Tokenization detail that mattered (Phase 2)

Exact character offsets are required for NER. Whitespace-only cutting recovered only **66.0%** of gold spans at exact boundaries. Cutting text into words plus single punctuation marks before BPE recovered **99.4%**, with each token's span recovered from its byte length (the `</w>` marker has zero width). An oracle test (gold spans to tags to perfect log-probabilities to decoder) confirms a strict macro-F1 ceiling of **0.994** from tokenization alone.

---

## Adaptation methods

Every method is hand-written for the model's `nn.Linear` layers, and each is tested so the adapted model starts exactly at the Phase-1 function.

| Method | Description |
|---|---|
| **Full fine-tuning** | All backbone weights trained |
| **LoRA** | `W' = W₀ + (α/r)·B·A` (rank 32, α 64 in Phases 2 and 3; rank 8, α 16 elsewhere) |
| **LoRMA** *(compulsory method)* | Multiplicative update `W' = (I + (α/r)·B·A)·W₀`, computed as `z = W₀x`, `y = z + (α/r)·B(Az)` |
| **Houlsby adapters** | Bottleneck adapters inserted into blocks |
| **Prefix tuning** | 32 key/value vectors per layer |
| **Soft prompts** | 16 virtual input tokens |
| **BitFit** | Bias-only training |
| **QLoRA** | 4-bit NF4-quantized backbone with LoRA (implemented via `NF4Linear`) |
| **Knowledge distillation** | 8-layer teacher to 4-layer student (used with Arch 4) |

Which methods each phase's notebooks expose through the `METHOD` setting:

| Phase | `METHOD` options |
|:---:|---|
| 2 | `lorma`, `lora`, `adapter`, `bitfit`, `partial`, `full` |
| 3 | `lorma`, `lora`, `qlora`, `prefix`, `full` |
| 4 | `lorma`, `lora`, `adapter`, `soft_prompt`, `full` |
| 5 | `soft_prompt`, `adapter`, `lora`, `lorma` (or `all` to run each in turn) |
| 6 | `soft_prompt`, `adapter`, `lora`, `lorma` |
| 7 | `METHODS = ['soft_prompt', 'adapter', 'lora', 'lorma']`, with an `ABLATION_METHOD` for the ablations |

---

## Results in detail

### Phase 1: baseline vs. modified

| Arch | Variant | Params (M) | Steps | Minutes | Val PPL | BS-F1 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | baseline | 158.7 | 762 | 223 | 21.18 | 0.8299 |
| 1 | **modified** | 145.0 | 1525 | 720 | **10.40** | **0.8449** |
| 2 | baseline | 158.7 | 762 | 247 | 20.35 | 0.8018 |
| 2 | modified | 145.0 | 762 | 771 | 21.27 | 0.8399 |
| 3 | baseline | 165.8 | 1525 | 253 | 22.98 | 0.8189 |
| 3 | modified | 154.3 | 1525 | 654 | 20.23 | 0.8299 |
| 4 | baseline | 172.8 | 1525 | 141 | 22.58* | 0.8086 |
| 4 | modified | 189.4 | 6103 | 416 | 12.02* | 0.8238 |

\* Arch 4 perplexity is measured on the continuation only, using a quick 128-sequence validation at step 1400, so it is **not comparable** with the other rows.

> **Caveat.** Only Arch 2 is a clean baseline-vs-modified comparison (both ran 762 steps). In Arch 1 and Arch 4 the modified model ran 2× and 4× more optimizer steps, so part of its perplexity advantage may come from step count rather than architecture. Arch 3 has no training logs for the baseline. These gains are indicative only.

### Best leaderboard score per phase and architecture

| Phase | Metric | Arch 1 | Arch 2 | Arch 3 | Arch 4 |
|:---:|---|---|---|---|---|
| 1 | BS-F1 | **0.8449** | 0.8399 | 0.8299 | 0.8238 |
| 2 | strict macro-F1 | **0.5610** LoRMA | 0.5501 LoRMA | 0.4911 LoRA | 0.2959 KD+LoRMA |
| 3 | macro-F1 | **0.5102** LoRA | 0.4866 prefix | 0.4936 LoRA | 0.3627 KD, full |
| 4 | macro-F1 | **0.7492** LoRMA | 0.6069 LoRA | 0.5060 adapter | 0.5511 soft prompt |
| 5 | macro-F1 | **0.3559** LoRMA | 0.3483 LoRA | 0.3376 LoRA | 0.2272 LoRA |
| 6 | BS-F1 | **0.8437** LoRA | 0.8280 LoRMA | 0.8377 adapter | 0.7956 soft prompt |
| 7 | micro-F1@7 | **0.2304** LoRA | 0.2085 adapter | 0.2120 LoRMA | 0.0118 soft prompt |

### Adaptation methods: Phases 2 and 3 (leaderboard)

| Phase | Arch | Full | LoRA | LoRMA | Prefix | Other |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 2 | 1 | 0.5393 | 0.5530 | **0.5610** | – | – |
| 2 | 2 | 0.5206 | 0.5490 | **0.5501** | – | – |
| 2 | 3 | – | **0.4911** | 0.4885 | – | BitFit 0.4742 |
| 2 | 4 | – | – | – | – | KD+LoRMA 0.2959 |
| 3 | 1 | 0.4911 | **0.5102** | 0.5061 | 0.4416 | – |
| 3 | 2 | 0.4551 | 0.4574 | 0.4700 | **0.4866** | – |
| 3 | 3 | 0.4678 | **0.4936** | 0.4621 | n/a | – |
| 3 | 4 (KD) | **0.3627** | 0.2965 | 0.2419 | n/a | – |

### Phase 7 retrieval ablation (Arch 1, dev F1@7)

| Variant | F1@7 |
|---|:---:|
| Pairwise margin loss | **0.286** |
| Default (mean pooling, 4 chunks, cosine, InfoNCE, hard negatives) | 0.265 |
| In-batch negatives only (29% faster to train) | 0.262 |
| Last-token pooling | 0.251 |
| Dot-product similarity | 0.228 |
| One chunk instead of four (a quarter of the training time) | 0.220 |

Each notebook also includes phase-specific analysis sections: error analysis and per-class breakdowns (Phases 2 and 3), threshold, calibration and "which part of the judgment matters" studies (Phase 4), label co-occurrence and window-attention analysis (Phase 5), coverage and redundancy analysis (Phase 6), and pooling, similarity and negative-sampling ablations (Phase 7).

---

## Key findings and lessons

1. **Backbone quality drove the ranking.** Arch 1 was best of our four systems in all seven phases, with the largest margin on the long-document task (Phase 4: 0.7492 vs. 0.607 for the next system). The encoder-decoder was last in six of seven phases. Its encoder only sees 768 tokens and is trained to model the continuation, so its encoder states appear to be a weak basis for classification. Some Arch 3 and 4 downstream runs also used earlier, under-trained Phase-1 checkpoints.
2. **Parameter-efficient methods matched or beat full fine-tuning** on every small dataset (Phases 2 and 3) while training roughly 0.4 to 7% of the parameters. LoRA and LoRMA differ by under 0.01 in most comparisons, which is within the noise of such small test sets.
3. **Pre-training loss is a poor guide to the generation metric.** Halving perplexity (Arch 1) moved BERTScore by only +0.015, while Arch 2 gained +0.038 at slightly worse perplexity.
4. **In Phase 4 the outcome sits at the end of the judgment.** Reading only the last 1,024 tokens (0.573 macro-F1) beats the first 1,024 (0.523), and learned attention concentrates on the last fifth of the document. A TF-IDF baseline (0.695 on dev) beat every architecture except Arch 1.
5. **NER overfits quickly.** Training loss falls to about 0.003 while validation F1 plateaus after epoch 8. BIO-constrained Viterbi decoding gave +0.045 over greedy, and the confidence threshold added about +0.02.
6. **Summarization fell below the lead baseline** on ROUGE-L (0.175 to 0.194 vs. 0.266), largely because of the 256-token generation cap against roughly 780-word reference summaries.
7. **Record the exact checkpoint every downstream result used.** An early Arch 3 checkpoint (step 600, val loss 3.58) was used in Phase 2 before the model was retrained to step 1400 (val loss 2.83).
8. **Plan controlled ablations early** (equal steps, batch size and tokens). Late comparisons are confounded.

**Failure cases documented:** the Phase 5 adapter run for Arch 4 produced NaN losses from epoch 1 (macro-F1 0.0002), and the Phase 4 soft-prompt model for Arch 4 collapsed to predicting class 0 until the decision threshold was tuned (0.333 to 0.568).

---

## Getting started

### Environment

The notebooks were developed and run on **Kaggle** with **2× Tesla T4 GPUs**, fp16 autocast and 12-hour session limits (Python 3). They use absolute `/kaggle/input/...` and `/kaggle/working/...` paths. Running locally or on another platform means adapting those paths.

**Python dependencies** (installed inside the notebooks where needed):

```
torch  numpy  tqdm  matplotlib  pandas  scikit-learn
huggingface_hub  kagglehub  ijson
bert_score  rouge_score
```

### 1. Prerequisites

| Requirement | Details |
|---|---|
| **Hugging Face token** | Needed to download the course dataset. Store it as a Kaggle secret named **`HF`**, or set the `HF_TOKEN` environment variable (Phases 2 to 4 also read the environment variable) |
| **Tokenizer** | Loaded with `kagglehub.dataset_download('kratikasum/tokenizer')`. A pre-tokenized corpus (`bpe_tokenized_txt.pkl`) is also searched for under `/kaggle/input` |
| **Phase-1 checkpoints** | Phases 2 to 7 need the Phase-1 `best.pt` or `last.pt` for the architecture they use, attached as a Kaggle dataset (these weights are **not stored in this repo**) |

### 2. Run Phase 1 (pre-training)

1. Open `Phase_1/Arch_N.ipynb` on Kaggle with the GPU T4 ×2 accelerator.
2. In the configuration cell, set `VARIANT = 'baseline'` or `'modified'`, and give ablation runs their own `RUN_TAG` so they don't share a `resume.pt`.
3. Run all cells. The notebook builds the train/validation token buffers (source-mixed), trains with periodic checkpointing, and picks the best checkpoint by validation perplexity.
4. If a session is interrupted, attach the saved output as an input dataset and rerun. Training resumes from `resume.pt`.
5. The last section generates continuations for the 1,000 test prompts and writes `prediction.json` and `submission.zip`.

### 3. Run Phases 2 to 7 (downstream tasks)

1. Attach the Phase-1 checkpoint for the matching architecture, plus the tokenizer dataset.
2. Add the `HF` secret.
3. Set `METHOD` (or `METHODS` in Phase 7) in the configuration cell. See [Adaptation methods](#adaptation-methods).
4. Run all cells. Each notebook trains, validates on its local dev split, produces the task-specific analysis, and writes a submission file plus saved weights and results.

**How checkpoints are located** (differs slightly by phase):

| Phase | Mechanism |
|:---:|---|
| 1 | `CKPT` path set in the config cell (Arch 1 and 2 point to a Kaggle dataset; Arch 3 and 4 default to `None`) |
| 2, 3, 4 | `CKPT = ''` triggers a search of `/kaggle/input/**/*.pt`, filtered by `CKPT_HINT` (e.g. `'arch1'`) |
| 5 | `CKPT_PATH = None` triggers a search of `/kaggle/input` and `/kaggle/working` |
| 6 | Hard-coded `CKPT` path to a Kaggle dataset, so edit it to your own |
| 7 | `glob` for `best.pt` (Arch 4: `last.pt`) under `/kaggle/input` |

Where a notebook expects a specific dataset path, edit that path to point at your own attached checkpoint dataset.

---

## Notebook anatomy and configuration knobs

Every notebook follows the same layout, which keeps the 28 notebooks easy to compare:

1. **Setup**: imports, Hugging Face login, GPU check
2. **Configuration**: the one cell you normally edit (`METHOD`, `ARCH_NAME`, `SEED = 42`, paths, learning rates, batch sizes, `CKPT` gradient-checkpointing flag)
3. **Tokenizer**: the shared byte-level BPE class and loader, with a quick encode/decode test
4. **Data loading and statistics**
5. **Model code** (hand-written, identical to Phase 1): `Config`, `RMSNorm`, `Rotary`, `MLP`, `KVCache`, `Attention`, `LMHead`, `Block`, `DecoderLM` (Arch 1 to 3) or `EncDecLM` (Arch 4). Arch 3 adds `LinearAttentionCore` and `LinearState`; Arch 4 adds `CrossAttention` and `DecBlock`
6. **Adaptation modules**: `LoRALinear`, `LoRMALinear`, `BottleneckAdapter`, `SoftPrompt`, `PrefixAttention`, `NF4Linear` (the set varies by phase)
7. **Task model and training loop**, with early stopping or best-step selection
8. **Evaluation, error analysis and submission writer**
9. **Save weights and results**

Useful switches:

| Setting | Where | Effect |
|---|---|---|
| `VARIANT` | Phase 1 | `'baseline'` or `'modified'` architecture |
| `RUN_TAG` | Phase 1 | Separate output folder and `resume.pt` for ablation runs |
| `METHOD` | Phases 2 to 6 | Selects the adaptation method |
| `CKPT` (bool) | all | Gradient checkpointing: saves GPU memory at roughly 30% slower speed |
| `SMOKE` | Phase 5 | Quick test on a tiny subset |
| `SEED` | all | Fixed at 42 for repeatability |

---

## Known limitations

- **Compute-limited pre-training.** Two T4 GPUs and 12-hour sessions allowed one epoch over 200 to 400M tokens, so Phase-1 models are far from converged (validation loss was still falling at the last step).
- **Confounded baseline-vs-modified comparisons** for Arch 1 and Arch 4 (different step counts and batch sizes), and no baseline logs for Arch 3.
- **Small local validation sets** (8 NER documents, 5+5 role documents, 351 summaries, 118 retrieval queries). Local and leaderboard scores diverged (e.g. Arch 1 NER: 0.606 local vs. 0.539 leaderboard), so differences under about one point are not meaningful.
- **No multiple seeds.** All runs use seed 42.
- **Mixed Phase-1 checkpoints.** Some Arch 3 and 4 downstream runs used earlier Phase-1 checkpoints than the final ones.
- **Metrics do not measure legal correctness.** BERTScore and ROUGE reward similarity to one reference and do not check that a cited section, party or holding is right. Strict span-F1 gives zero credit for an off-by-one-character boundary. Any real deployment would need human review and faithfulness checks.
- **Not a pipeline.** Each task is fine-tuned separately, so there is no error propagation, but also no sharing of structure across phases (e.g. NER spans as features for statute identification).
- **Repository contents.** Some notebooks mention a helper, `q4_peft.py` (the PEFT module from an earlier assignment, originally written for GPT-2 `Conv1D` layers). It is not included here, and the notebooks re-implement the techniques for `nn.Linear` internally. Model weights, tokenizer files and the dataset are also not included.

---

## Future work

1. Re-run the Phase-1 baseline/modified pairs with identical steps, batch size and tokens, across three seeds.
2. Move Phase 7 from the first four chunks to full-document retrieval and add a cross-encoder re-ranker.
3. Raise the summarization generation cap to about 800 tokens with chunk-then-merge, and compare against an extractive lead-plus-selection hybrid.
4. Add a span-level CRF and cross-validation over documents for Phase 2.
5. Evaluate summary faithfulness with a statute-and-party consistency check.
6. Share structure across phases (e.g. NER spans as features for Phase 5).

---

## LLM usage declaration

An LLM (Claude) was used for: explaining concepts and papers (LoRMA, GLA, RetNet), brainstorming ablations and long-document strategies, debugging small shape and CUDA out-of-memory issues, the Phase-1 loss-plot script, and proofreading the report. It was **not** used for model architecture design, hyperparameter search, analysis of results, or report structure. See the final pages of the report for the full declaration.

---

## References

Selected references (the full bibliography is in the report):

- Zhang et al., *BERTScore: Evaluating Text Generation with BERT*, ICLR 2020
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- Bihany, Patel and Modi, *LoRMA: Low-Rank Multiplicative Adaptation for LLMs*, Findings of ACL 2025
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023
- Su et al., *RoFormer: Rotary Position Embedding*
- Ainslie et al., *GQA: Grouped-Query Attention*, EMNLP 2023
- Jiang et al., *Mistral 7B*, 2023
- Sun et al., *Retentive Network (RetNet)*, 2023
- Yang et al., *Gated Linear Attention Transformers*, ICML 2024
- Gloeckle et al., *Better & Faster LLMs via Multi-Token Prediction*, ICML 2024
- Dettmers et al., *QLoRA*, NeurIPS 2023
- Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019
- Li and Liang, *Prefix-Tuning*, ACL 2021; Lester et al., *Prompt Tuning*, EMNLP 2021
- Ben Zaken et al., *BitFit*, ACL 2022
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016
- van den Oord et al., *Representation Learning with Contrastive Predictive Coding (InfoNCE)*, 2018

---

## Acknowledgements

Course: **CS781, Advanced NLP: Large Language Models**, Department of Computer Science and Engineering, IIT Kanpur. Data and evaluation: the course capstone dataset ([Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone)) and Codabench leaderboard.
