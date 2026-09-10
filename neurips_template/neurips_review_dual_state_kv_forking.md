# NeurIPS 2026 Comprehensive Paper Review & Reference Audit

**Paper Title:** Dual-State KV Forking: Low-Overhead Semantic Interruption for Full-Duplex LLM Agents  
**Tracking / Submission ID:** NeurIPS 2026 Anonymous Submission  
**Primary Subject Areas:** Machine Learning Systems / Systems for ML (Inference, Serving, and Memory Architectures); Multi-Agent Systems; Conversational AI and Dialogue Modeling

---

## 1\. Executive Summary & Meta-Review

### 1.1 Overview of the Work

This paper addresses the fundamental architectural limitation of half-duplex communication in multi-agent Large Language Model (LLM) workflows. In conventional multi-agent frameworks, listening agents remain entirely passive until a speaking agent completes its full generation turn. While recent input streaming mechanisms mitigate end-of-turn Time-to-First-Token (TTFT) by progressively warming the Key-Value (KV) cache, they lack proactive floor control—leaving agents unable to halt hallucinations, resolve ambiguities early, or inject constraints mid-turn.

To overcome the prohibitive $O(N^2 \\cdot k)$ context re-prefill and latency overheads of invoking independent, stateless deciders at every streaming token chunk, the authors introduce **Dual-State KV Forking**:

1. **Persistent Base State:** Ingests the streaming token sequence once into a continuous KV cache ($T\_{base} \= \\sum k \= T$).  
2. **Ephemeral Assessor State:** At each chunk boundary ($k=5$), the Base cache is shallow-forked via pointer aliasing/copy-on-write into an isolated assessment branch, appending a 10-token decider prompt ($P\_{eval}$).  
3. **Constrained Binary Logit Gating:** A single forward pass evaluates logits over ${id(\\text{STOP}), id(\\text{CONTINUE})}$, executing a 1-step argmax decision without autoregressive chain-of-thought (CoT) generation.  
4. **Non-destructive Discard & Warm No-Op Path:** Benign continuations discard the ephemeral state, leaving the Base cache warm and intact for near-instantaneous response generation if the turn concludes naturally.

The authors evaluate the approach on an NVIDIA RTX 4070 Ti (4-bit quantization) across Gemma 4 12B, Llama 3.1 8B, and Qwen 3 4B on the FLEXI dialogue benchmark, Full-Duplex-Bench (FDB), and a 200-scenario progressive-clue floor-control benchmark (HANDRAISER).

### 1.2 Meta-Review Recommendation & Consensus Verdict

* **Consensus Recommendation:** **Borderline Reject / Weak Accept (Score: 5 / 10\)**  
* **Meta-Review Summary:**  
  The core systems insight—repurposing KV cache branching and copy-on-write primitives to turn passive streaming listeners into active conversational participants—is conceptually elegant, timely, and of significant practical interest to both the systems and multi-agent AI communities. The paper provides physical hardware measurements across multiple model families and demonstrates substantial reductions in redundant token ingestion.  
  However, both systems and empirical reviewers identified major methodological and technical concerns that prevent an outright accept in its current form:  
  1. **Uncharitable Baseline Comparisons:** The headline claim of cutting total tokens processed by 78.6% is benchmarked against an un-cached stateless decider that generates 15–35 CoT tokens at every 5-token chunk. The paper omits the fair and practical baseline—a stateful incremental listener or a prefix-cached decider—from its main continuous floor-control evaluation (Table 2).  
  2. **Contradictory Memory Cloning Claims:** The paper cites a 0.01–0.06 ms "deep-clone overhead" in vanilla PyTorch, which is physically impossible for hundreds of megabytes of KV tensors on consumer GPU memory bandwidth. This measurement reflects CPU-side pointer/metadata manipulation rather than tensor duplication.  
  3. **Benchmark Construct Validity:** HANDRAISER adapts Quizbowl trivia (adversarial entity guessing), which is an adversarial, information-monotonic game that does not reflect real-world cooperative multi-agent floor negotiation (e.g., constraint negotiation, debate, tool-calling preemption).  
  4. **Uncalibrated Binary Logit Gating:** Relying on raw uncalibrated logit differences ($z\_{STOP} \> z\_{CONTINUE}$) makes the decision boundary brittle to tokenization artifacts (leading spaces, subword splitting) and pre-training unigram frequency bias.  
  5. **Empirical Anomalies & Sample Size:** On Qwen 3 4B, Dual-State causes a 31.7% increase in consensus time due to premature false-positive interruptions. Furthermore, the 200-scenario benchmark lacks statistical error bars or significance testing for marginal 1.0%–2.5% accuracy deltas.

---

## 2\. Consolidated NeurIPS Review Scores

| Criterion | Score | Scale | Qualitative Assessment |
| :---- | :---: | :---: | :---- |
| **Overall Rating** | **5** | 1–10 | Borderline reject: strong motivation and systems concept, but hampered by baseline fairness, benchmark misalignment, and unverified memory claims. |
| **Reviewer Confidence** | **4** | 1–5 | High: reviewers possess deep expertise in LLM inference serving, KV cache management, and multi-agent evaluation. |
| **Soundness** | **2** | 1–4 | Fair: good physical profiling, but minor/moderate issues in baseline design, statistical rigor, and memory clone characterization. |
| **Presentation** | **3** | 1–4 | Good: well-written and logically organized; needs clearer distinction between metadata aliasing and data copying, and transparent baseline framing. |
| **Contribution** | **3** | 1–4 | Good: a creative, practical execution pattern bridging inference-serving primitives and full-duplex conversational agents. |

---

## 3\. Systems, Architecture & Methodological Evaluation

### 3.1 Technical Soundness of KV Forking & Tensor Management

* **The "0.01–0.06 ms Deep-Clone" Inconsistency:**  
  In Section 3 (lines 82–84), the text claims: *"measured as 0.01–0.06 ms deep-clone overhead in our PyTorch validation prototype, and zero-copy in production engines"*.  
  In PyTorch, a true deep copy (`tensor.clone()`) requires allocating a new memory buffer and copying physical tensor contents over PCIe/HBM. On an RTX 4070 Ti (\~504 GB/s theoretical bandwidth), copying a 100–500 MB KV cache takes several milliseconds (0.5–2.0 ms), not 10–60 $\\mu$s. The 0.01–0.06 ms figure measures only Python object instantiation or metadata view referencing (`tensor.detach()`). However, appending new prompt tokens ($P\_{eval}$) to an aliased tensor in vanilla PyTorch requires tensor reallocation or in-place modification that would corrupt the Base state. The authors must clarify the exact memory allocator mechanics used in their prototype.  
* **Serving Engine Integration Realities (vLLM / SGLang):**  
  In production engines utilizing PagedAttention or RadixAttention, forking is achieved by duplicating the *block table* (logical-to-physical block mapping) and incrementing block reference counts. Appending $P\_{eval}$ (10 tokens) allocates only 1–2 new physical pages under Copy-on-Write (CoW). The paper should formally frame Dual-State as a *block-table CoW operation* rather than conflating it with PyTorch tensor cloning.  
* **Causal Attention Compute Complexity:**  
  The paper asserts that cumulative prefill cost is linear ($T\_{base} \= \\sum k \= T$). While linear projection FLOPs ($W\_Q, W\_K, W\_V, W\_O$) scale linearly with tokens, **causal self-attention compute is strictly quadratic in sequence length** ($O(T^2)$). At chunk $m$, query tokens must attend over all past $m \\cdot k$ tokens: $$\\sum\_{m=1}^M m \\cdot k^2 \\approx \\frac{1}{2} T^2$$ Unless paired with an attention-sink or rolling-buffer eviction policy (e.g., StreamingLLM), attention compute will bottleneck real-time latency as dialogues grow long.

### 3.2 Logit Gating Robustness & Tokenization Vulnerabilities

* **Tokenizer Discrepancies:** Constraining the decision to ${id(\\text{STOP}), id(\\text{CONTINUE})}$ assumes that the model produces these exact token IDs at position $L\_p$. In modern tokenizers (BPE with 128k vocab in Llama 3.1, SentencePiece in Gemma, Qwen BPE), token IDs differ depending on preceding punctuation and whitespace (e.g., `" STOP"` vs. `"STOP"` vs. `":STOP"`).  
* **Uncalibrated Argmax Decision Boundary:** The decision rule $z\_{STOP} \> z\_{CONTINUE}$ represents an uncalibrated 0.5 probability threshold. If `"CONTINUE"` has a higher marginal unigram frequency in the pre-training corpus than `"STOP"`, raw logits are naturally skewed toward continuation. The absence of temperature scaling or a tunable decision margin $\\tau$ ($z\_{STOP} \- z\_{CONTINUE} \> \\tau$) prevents operators from tuning precision/recall trade-offs.

### 3.3 Hardware Constraints & Quantization Confounders

* Benchmarking exclusively on an RTX 4070 Ti with 4-bit quantization introduces quantization noise that alters logit dynamic ranges and tail probabilities. Furthermore, small chunk execution ($k=5$) with 4-bit weights is frequently launch-overhead and dequantization bound, which may obscure the true scaling characteristics of unquantized FP16/BF16 deployments.

---

## 4\. Empirical Evaluation & Multi-Agent Benchmark Analysis

### 4.1 Benchmark Construct Validity

* **Quizbowl vs. Collaborative Multi-Agent Dialogue:** HANDRAISER utilizes 200 trivia scenarios (100 AdvQA, 100 protobowl). Quizbowl is an adversarial, zero-sum game with monotonic clue progression (obscure to specific). Real-world multi-agent coordination (e.g., collaborative coding, planning, task delegation) features non-monotonic information flow and cooperative floor negotiation where interruptions serve to inject constraints, correct logic drifts, or halt tool loops.  
* **Arbitrary Scoring Formulation (Equation 1):**  
  The game score is defined as: $$S\_i \= \[(Solved\_i) \\cdot (B \+ \\alpha \\cdot \\text{SpeedBonus})\] \- \\sum \\beta$$ with $B=10, \\alpha=10, \\beta=0.25$. Setting the incorrect buzz penalty to $\\beta \= 0.25$ while rewarding early buzzing with up to 10 points creates an aggressive incentive to buzz prematurely. In production systems, false-positive interruptions impose severe conversation repair costs.

### 4.2 Cross-Model Performance Anomalies

* **The Qwen 3 4B Consensus Slowdown:** In Table 2, Dual-State reduces Qwen's Time-to-Halt (0.0934 s $\\to$ 0.0857 s), but causes Consensus Time to worsen from 3.0786 s to **4.0542 s (+31.7% slower)**. Because smaller models have lower semantic discrimination under zero-shot logit gating, Qwen suffers from premature, incorrect interruptions, trapping the agents in costly multi-attempt recovery loops.  
* **Missing Statistical Confidence:** With only 200 scenarios, the observed accuracy gains (+2.5% on Gemma, \+2.5% on Llama, \+1.0% on Qwen) represent a swing of only **2 to 5 questions** across the entire evaluation set. Without bootstrap confidence intervals or significance tests, these improvements cannot be distinguished from random noise.  
* **Selective Baseline Reporting in Table 2:** While Table 1 reports the "Independent \+ Logit Gating" baseline (isolating the effect of removing CoT reasoning), Table 2 compares Dual-State exclusively against the stateless reasoning decider. Much of the reported token savings and latency reduction in Table 2 stems from eliminating CoT reasoning tokens rather than from KV forking alone.

---

## 5\. Comprehensive Audit of Existing References

An exhaustive line-by-line verification was conducted across all 22 references in the paper:

### Detailed Reference Inventory & Verification Table

| Ref \# | In-Paper Citation | Verified Author List | Verified Title | Year | Venue / Status | Audit Verdict & Action |
| :---: | :---- | :---- | :---- | :---: | :---- | :---- |
| **1** | Agrawal et al. (2024) | Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee | Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve | 2024 | OSDI '24, pp. 1127–1143 | **Completely Correct** |
| **2** | Defossez et al. (2024) | Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Édouard Grave, Neil Zeghidour | Moshi: a speech-text foundation model for real-time dialogue | 2024 | arXiv:2410.00037 | **Minor Correction**: Restore French accents (*Défossez, Mazaré, Amélie, Pérez, Jégou, Édouard*). |
| **3** | Gemma Team (2026) | Gemma Team (Google DeepMind) | Gemma 4 technical report | 2026 | arXiv:2607.02770 | **Completely Correct** |
| **4** | Gim et al. (2024) | In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, Lin Zhong | Prompt Cache: Modular attention reuse for low-latency inference | 2024 | MLSys, vol. 6, pp. 341–353 | **Completely Correct** |
| **5** | Grattafiori et al. (2024) | Llama Team / Abhimanyu Dubey et al. | The Llama 3 herd of models | 2024 | arXiv:2407.21783 | **Minor Correction**: Citing "Grattafiori et al." is an arXiv metadata artifact; standard citation is *Abhimanyu Dubey et al.* or *Llama Team*. |
| **6** | Kwon et al. (2023) | Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica | Efficient memory management for large language model serving with PagedAttention | 2023 | SOSP '23, pp. 611–626 | **Completely Correct** |
| **7** | Leviathan et al. (2023) | Yaniv Leviathan, Matan Kalman, Yossi Matias | Fast inference from transformers via speculative decoding | 2023 | ICML '23, PMLR 202, pp. 19274–19286 | **Completely Correct** |
| **8** | Li et al. (2023) | Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem | CAMEL: Communicative agents for "mind" exploration of large language model society | 2023 | NeurIPS 36, pp. 51991–52008 | **Completely Correct** |
| **9** | Lin et al. (2022) | Ting-En Lin, Yuchuan Wu, Fei Huang, Luo Si, Jian Sun, Yongbin Li | Duplex conversation: Towards human-like interaction in spoken dialogue systems | 2022 | KDD '22, pp. 3438–3447 | **Completely Correct** |
| **10** | Lin et al. (2025) | Guan-Ting Lin, Jiachen Lian, Tingle Li, Qirui Wang, Gopala Anumanchipalli, Alexander H. Liu, Hung-yi Lee | Full-Duplex-Bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities | 2025 | Proc. IEEE ASRU 2025 / arXiv:2503.04721 | **Completely Correct** |
| **11** | Nguyen et al. (2023) | Tu Anh Nguyen, Eugene Kharitonov, Jade Copet, Yossi Adi, Wei-Ning Hsu, Ali Elkahky, Paden Tomasello, Robin Algayres, Benoît Sagot, Abdelrahman Mohamed, Emmanuel Dupoux | Generative spoken dialogue language modeling | 2023 | TACL, 11:250–266 | **Completely Correct** (Minor: restore circumflex on *Benoît*). |
| **12** | Rodriguez et al. (2019) | Pedro Rodriguez, Shi Feng, Mohit Iyyer, He He, Jordan Boyd-Graber | Quizbowl: The case for incremental question answering | 2019 | arXiv:1904.04792 | **Completely Correct** |
| **13** | Sacks et al. (1974) | Harvey Sacks, Emanuel A. Schegloff, Gail Jefferson | A simplest systematics for the organization of turn-taking for conversation | 1974 | Language, 50(4):696–735 | **Completely Correct** |
| **14** | Skantze (2021) | Gabriel Skantze | Turn-taking in conversational systems and human-robot interaction: A review | 2021 | Computer Speech & Language, 67:101178 | **Completely Correct** |
| **15** | Wallace et al. (2019) | Eric Wallace, Pedro Rodriguez, Shi Feng, Ikuya Yamada, Jordan Boyd-Graber | Trick me if you can: Human-in-the-loop generation of adversarial examples for question answering | 2019 | TACL, 7:387–401 | **Completely Correct** |
| **16** | Wu et al. (2023) | Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, Chi Wang | AutoGen: Enabling next-gen LLM applications via multi-agent conversations | 2024 | Published in COLM 2024 / arXiv:2308.08155 | **Minor Correction**: Update publication venue from arXiv preprint to peer-reviewed *COLM 2024*. |
| **17** | Xiao et al. (2024) | Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis | Efficient streaming language models with attention sinks | 2024 | ICLR 2024 | **Completely Correct** |
| **18** | Yang et al. (2025) | An Yang et al. (Qwen Team) | Qwen3 technical report | 2025 | arXiv:2505.09388 | **Completely Correct** |
| **19** | Yngve (1970) | Victor H. Yngve | On getting a word in edgewise | 1970 | Proc. 6th Regional Meeting of Chicago Linguistic Society, pp. 567–578 | **Completely Correct** |
| **20** | Yu et al. (2022) | Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun | Orca: A distributed serving system for transformer-based generative models | 2022 | OSDI '22, pp. 521–538 | **Completely Correct** |
| **21** | Zhang et al. (2024) | Xinrong Zhang, Yingfa Chen, Shengding Hu, Xu Han, Zihang Xu, Yuanwei Xu, Weilin Zhao, Maosong Sun, Zhiyuan Liu | Beyond the turn-based game: Enabling real-time conversations with duplex models | 2024 | EMNLP '24, pp. 11543–11557 | **Completely Correct** |
| **22** | Zheng et al. (2024) | Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng | SGLang: Efficient execution of structured language model programs | 2024 | NeurIPS 37 | **Completely Correct** |

### Critical Omission Identified in the Text:

* **FLEXI Benchmark (Missing from References):**  
  In lines 47, 105, 109, and Table 1, the authors prominently report evaluation results on the **FLEXI** benchmark (400 human dialogue transcripts). However, **FLEXI is completely missing from the reference list**. It must be added:  
  * *Yuan Ge et al., "FLEXI: Benchmarking Full-duplex Human-LLM Speech Interaction", arXiv preprint arXiv:2509.22243, 2025\.*

---

## 6\. Recommended Newer & Relevant Literature (2024–2026)

*Note: Per the user request, these references are presented here for review and approval before being incorporated into the paper.*

### Category 1: Full-Duplex Dialogue & Real-Time Turn-Taking (2024–2026)

2. **FLEXI Benchmark Citation (Essential):**  
   * *Citation:* Yuan Ge, Saihan Chen, Jingqi Xiao, Xiaoqian Liu, Tong Xiao, Yan Xiang, et al. *FLEXI: Benchmarking Full-duplex Human-LLM Speech Interaction*. arXiv preprint arXiv:2509.22243, 2025\.  
   * *Why Include:* Formally grounds the primary 400-transcript benchmark utilized in Table 1 and Section 4\.  
3. **Mini-Omni2 (Duplex Token Gating):**  
   * *Citation:* Zhifei Xie, Changqiao Wu, et al. *Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities*. arXiv preprint arXiv:2410.11190, 2024\.  
   * *Why Include:* Demonstrates native token-level interrupt flags (`{irq}` vs. `{n-irq}`) for concurrent streaming interruption, providing a direct point of comparison for text-based decider gating.  
4. **Full-Duplex-Bench-v2 (Multi-Turn Evaluation):**  
   * *Citation:* Guan-Ting Lin, Shih-Yun Ke, Gopala Anumanchipalli, Alexander H. Liu, Hung-yi Lee. *Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner*. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2026\.  
   * *Why Include:* Extends the authors' existing FDB benchmark (Ref 10\) to dynamic multi-turn interactions, directly motivating continuous floor-control mechanisms.

### Category 2: KV Cache Optimization & Tree-Branching Attention (2024–2026)

5. **DeFT: Decoding with Flash Tree-Attention (ICLR 2025):**  
   * *Citation:* Jinwei Yao, Kaiqi Chen, Kexun Zhang, Jiaxuan You, et al. *DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference*. In *Proceedings of the Thirteenth International Conference on Learning Representations (ICLR)*, 2025\.  
   * *Why Include:* Solves the exact GPU I/O memory bandwidth bottlenecks when branching from shared KV prefixes, directly validating Appendix F's serving-engine mapping.  
6. **ArborKV: Structure-Aware KV Cache Management (2026):**  
   * *Citation:* *ArborKV: Structure-Aware KV Cache Management for Scaling Tree-based LLM Reasoning*. arXiv preprint arXiv:2605.22106, 2026\.  
   * *Why Include:* Provides concrete memory layout techniques for ephemeral tree-branching in KV caches, demonstrating how to prevent memory fragmentation during high-frequency forking.  
7. **CacheBlend: Fast LLM Serving with Knowledge Fusion (EuroSys 2025):**  
   * *Citation:* Jiayi Chen et al. *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*. In *Proceedings of the European Conference on Computer Systems (EuroSys)*, 2025\.  
   * *Why Include:* Explores non-destructive cache reuse during continuous streaming, substantiating the authors' claims regarding warm no-op TTFT preservation.

### Category 3: Multi-Agent Coordination & Interruption Protocols (2024–2026)

8. **Learning to Interrupt in Multi-Agent Communication (2026):**  
   * *Citation:* Danqing Wang, Da Yin, Ruta Desai, Lei Li, Asli Celikyilmaz, Ansong Ni. *Learning to Interrupt in Language-based Multi-agent Communication*. arXiv preprint arXiv:2604.06452, 2026\.  
   * *Why Include:* Investigates the communicative policies of *when* agents should interrupt; Dual-State KV Forking provides the runtime systems engine to execute these policies without latency penalties.  
9. **Speculative Actions in Multi-Agent Systems (2025):**  
   * *Citation:* Naimeng Ye, Arnav Ahuja, Georgios Liargkovas, Yunan Lu, et al. *Speculative Actions: A Lossless Framework for Faster Agentic Systems*. arXiv preprint arXiv:2510.04371, 2025\.  
   * *Why Include:* Demonstrates the cost of wasted re-prefills and token churn during collaborative agent turns, highlighting the exact computational bottleneck that Dual-State resolves.

---

## 7\. Actionable Rebuttal & Revision Roadmap for Authors

1. **Benchmark Against a Stateful / Prefix-Cached Baseline:**  
   Update Table 2 to compare Dual-State against an independent decider with prefix caching or an incremental stateful listener. Reframe the narrative: the core systems achievement is *eliminating redundant ingestion passes and cutting KV memory by 50% relative to stateful deciders*, rather than claiming 78% savings against an un-cached stateless model.  
2. **Accurately Characterize Memory Allocator Mechanics:**  
   Replace the terminology "deep-clone overhead in PyTorch" with "block-table duplication overhead in paged KV memory". Provide profiling data for physical page table duplication and Copy-on-Write page allocation.  
3. **Calibrate the Decision Boundary:**  
   Formulate the gating criterion as $z\_{STOP} \- z\_{CONTINUE} \> \\tau$. Provide ROC/PR curves showing how varying $\\tau$ balances False Interruption Rate against Time-to-Halt.  
4. **Expand Evaluation to Pragmatic Multi-Agent Tasks:**  
   Supplement the Quizbowl trivia benchmark with a collaborative multi-agent task (e.g., interactive coding with constraint changes, or multi-agent debate with factual correction) to demonstrate genuine conversational floor control.  
5. **Statistical Reporting:**  
   Provide 95% bootstrap confidence intervals across benchmark runs and conduct paired significance tests to establish that accuracy gains are statistically robust.

&nbsp;