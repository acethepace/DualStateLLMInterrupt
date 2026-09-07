# Dual-State LLM Interruption Architecture

[![NeurIPS 2026 Submission](https://img.shields.io/badge/Paper-NeurIPS%202026-blue.svg)](neurips_paper.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()
[![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

This repository contains the official implementation, empirical benchmarks, evaluation datasets, and paper preprint for:

> **Dual-State LLM Interruption Architecture: Enabling Real-Time Full-Duplex Multi-Agent Communication**  
> *Under review at NeurIPS 2026*  
> **PDF Preprint:** [`neurips_paper.pdf`](neurips_paper.pdf) | **LaTeX Source:** [`neurips_paper.tex`](neurips_paper.tex) | **Author Rebuttal:** [`REBUTTAL_NEURIPS.md`](REBUTTAL_NEURIPS.md)

---

## 📌 Overview

In conventional multi-agent interactions, a listening agent must wait passively for the speaking agent to finish speaking before processing the message ($O(N)$ turn-taking delay), or repeatedly re-prefill the entire dialogue context from scratch at every incoming chunk ($O(N^2 \cdot k)$ compute crisis).

The **Dual-State LLM Interruption Architecture** overcomes this dilemma through a parallel, zero-latency inference mechanism:
1. **Base State (Continuous Stream Ingestion):** Progressively updates an incremental Key-Value (KV) cache as the speaker streams tokens ($O(1)$ token decode cost).
2. **Assessor State (KV-Cache Forking):** At discrete chunk intervals ($k=5$ tokens), the KV cache is forked via pointer aliasing/shallow cloning ($0.01\text{--}0.06\text{ ms}$ overhead).
3. **1-Token Logit Gating:** A short decider suffix ($L_p = 10$) evaluates floor control via binary logit gating over $\{\texttt{STOP}, \texttt{WAIT}\}$, bypassing multi-token autoregressive decoding.
4. **Warmed KV Reuse (No-Op):** If no interruption occurs, the warmed Base KV cache is immediately repurposed for generation, reducing Time-to-First-Token (TTFT) by up to **$2.86\times$**.

```
Speaker Stream ---> [Chunk C_m] 
                          |
                          v
         +---------------------------------+
         | Base State: Ingest into KV      | ===(No-Op)===> Response Generation
         | K_{0:t}, V_{0:t}                |                (TTFT up to 2.86x faster)
         +---------------------------------+
                          |
              Shallow Fork (0.02 ms)
                          v
         +---------------------------------+
         | Assessor State: Forked Cache    |
         | + Suffix Lp ("Classification:") |
         +---------------------------------+
                          |
                 Logit Gating Layer
                          |
               +----------+----------+
               |                     |
           P(WAIT)                P(STOP)
               |                     |
         Continue Stream      HALT SPEAKER (Floor-Taking)
```

---

## 🚀 Key Empirical Results

All evaluations were benchmarked on physical NVIDIA GPU hardware across **Gemma 4 12B**, **Llama 3.1 8B**, and **Qwen 3 4B**:

| Benchmark | Metric | Baseline (Independent Decider) | Dual-State (Proposed) | Improvement |
|:---|:---|:---:|:---:|:---:|
| **Token Efficiency** | Redundant Context Prefill | 576.2 tokens / scenario | **0.0 tokens** | **-100.0%** |
| **Token Efficiency** | Total Tokens Processed | 835.5 tokens / scenario | **207.1 tokens** | **-75.2% Net Compute** |
| **TTFT Speedup** | No-Op First Token ($L=1024$) | 234.65 ms (Llama) / 447.30 ms (Gemma) | **82.18 ms** / **229.46 ms** | **Up to 2.86$\times$ faster** |
| **Floor-Taking Latency** | Time-to-Halt (Gemma 4 12B) | 0.2205 s | **0.1856 s** | **-15.8% Latency** |
| **Consensus Speed** | Consensus Time (Qwen 3 4B) | 0.8540 s | **0.8024 s** | **+6.0% Speedup** |
| **Semantic Accuracy** | FLEXI Interruption F1 | 92.25% | **97.25%** | **State-of-the-Art** |
| **Dynamic Multi-Agent**| Full-Duplex-Bench Precision | 90.50% | **100.0%** | **Perfect Precision** |
| **KV Memory Footprint**| High Concurrency ($B=64$) | 5,816.2 MB | **2,936.2 MB** | **-49.5% Memory Saved** |

---

## 📂 Repository Structure

```
.
├── neurips_paper.pdf                        # Compiled 9-page NeurIPS 2026 paper preprint
├── neurips_paper.tex                        # LaTeX source code for paper
├── REBUTTAL_NEURIPS.md                      # Official point-by-point NeurIPS rebuttal document
├── AGENTS.md                                # Architectural principles & experimental logs
├── run_full_duplex_handraiser_official.py   # True concurrent full-duplex HANDRAISER runner
├── run_reviewer_experiments.py              # Chunk size sweep, linear probe, & concurrency scaling
├── run_noop_ttft_experiment.py              # Physical No-Op TTFT & KV-fork latency benchmark
├── run_threshold_sweep.py                   # Gating threshold sensitivity sweep (tau in [0.1, 0.9])
├── quizbowl_data/                           # Curated adversarial clue scenarios (AdvQA & Protobowl)
├── results/                                 # Raw experimental JSONs across all model architectures
└── notebooklm/                              # Synchronized files for NotebookLM research integration
```

---

## 🛠️ Quickstart

### 1. Installation

```bash
git clone https://github.com/acethepace/DualStateLLMInterrupt.git
cd DualStateLLMInterrupt

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # torch, transformers, bitsandbytes, accelerate
```

### 2. Run True Full-Duplex Concurrent Streaming Benchmark

```bash
python run_full_duplex_handraiser_official.py
```

### 3. Run Reviewer Ablations (Chunk Stride $k$, Linear Probe, Memory Scaling)

```bash
python run_reviewer_experiments.py
```

### 4. Recompile Paper PDF

```bash
cd neurips_template
pdflatex -interaction=nonstopmode neurips_paper.tex
```

---

## 📜 Citation

```bibtex
@article{dualstate2026,
  title={Dual-State LLM Interruption Architecture: Enabling Real-Time Full-Duplex Multi-Agent Communication},
  author={Anonymous Authors},
  journal={Preprint under review at NeurIPS},
  year={2026}
}
```
