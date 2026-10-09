# Legal Language Model from Scratch (CS-781 Capstone)

A seven-phase project to build a language model for Indian legal text, starting from nothing: custom tokenizer, hand-written architectures, pretraining on the [Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone) corpus, and later adaptation and evaluation.

**Current status: Phase 1 of 7 (pretraining and architecture selection) is complete.** Phases 2 to 7 build on the model selected here.

| Phase | Focus | Status |
|---|---|---|
| 1 | Legal Language Model Pretraining | Done |
| 2 | Legal Named Entity Recognition (L-NER) | In Progress |
| 3 | Rhetorical Role Prediction (RR) | Not started |
| 4 | Court Judgment Prediction (CJPE) | Not started |
| 5 | Legal Statute Identification (LSI) | Not started |
| 6 | Abstractive Summarization | Not started |
| 7 | Prior Case Retrieval (PCR) | Not started |

## Phase 1 summary

**Task.** Train a 50M to 300M parameter language model from scratch (no pretrained weights) on legal text (Supreme Court, High Court, tribunal and Acts documents) and pick the model(s) to carry forward. Models are scored by BERTScore F1 on 128-word continuations of 1000 test prompts, and by validation perplexity.

**What is here.** Four hand-written architectures, each with a baseline and a modified variant, sharing one tokenizer and data pipeline:

| # | Architecture | Baseline | Modified |
|---|---|---|---|
| 1 | LLaMA-style decoder-only | RMSNorm, RoPE, SwiGLU, MHA, 12 layers | + GQA, QK-norm, value residual, multi-token prediction, source embedding, 17 layers |
| 2 | Local/global hybrid decoder | Sliding window 512 in every layer | + windows growing 128→512, global NoPE layer every 4th layer, GQA, QK-norm, MTP |
| 3 | Linear-attention / recurrent | RetNet-style retention | + input-dependent decay, short causal conv, L2-normed q/k, softmax layer every 4th layer |
| 4 | Encoder-decoder | 6 enc + 6 dec, cross-attention every layer | 8 enc + 9 dec, cross-attention every 2nd layer, downsampled encoder output, GQA, QK-norm, MTP |

Shared components: byte-level BPE tokenizer with a word-end marker (48,000 tokens), 2048-token context, KV cache plus a recurrent state for fast generation.

### Results

| Run | Params (M) | Train tokens | Train min | Val PPL | BERTScore F1 |
|---|---|---|---|---|---|
| A1 baseline | 158.7 | 400M | 223 | 21.18 | 0.8299 |
| **A1 modified** | 145.0 | 400M | 720 | **10.40** | **0.8449** |
| A2 baseline | 158.7 | 400M | 247 | 20.35 | 0.8018 |
| A2 modified | 145.0 | 400M | 771 | 21.27 | 0.8399 |
| A3 baseline | 152.4 | 400M | 400 | 40 | 0.7991 |
| A3 modified | 149 | 400M | 439 | 36.7 | 0.8181 |
| A4 baseline | 172.8 | 400M | 142 | 22.58* | 0.8086 |
| A4 modified | 189.4 | 400M | 416 | 12.02* | 0.8238 |
