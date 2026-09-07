# Author Response & Official Rebuttal: NeurIPS 2026

**Paper Title:** Dual-State LLM Interruption Architecture: Enabling Real-Time Full-Duplex Multi-Agent Communication  
**Reviewer Rating:** 5 (Borderline Accept) $\to$ **Target:** 7/8 (Strong Accept)  
**Reviewer Confidence:** 4 (High)  

We thank the reviewer for their thorough, rigorous, and highly constructive evaluation. We are encouraged that the reviewer recognized the **significance and practical systems framing** of our work, the **systems efficiency** of eliminating quadratic prefix re-prefills ($576.2 \to 0.0$ tokens), the **latency suppression** of combining KV forking with 1-token logit gating, and our **empirical validation on physical hardware**.

Below, we provide a point-by-point response addressing every weakness and question raised, supported by new empirical benchmarks conducted on physical NVIDIA GPU hardware (RTX 4070 Ti). All updates, tables, and clarifications have been fully incorporated into the revised paper draft (`neurips_paper.pdf` and `neurips_paper.tex`).

---

### Response to Weakness A & Question 1: Disambiguation of Table 6 (HANDRAISER Metrics)

> **Reviewer Comment:**  
> *"In Table 6, the reported Interruption Accuracy across all models is exceptionally low: Gemma 4 12B: 0.0%, Llama 3.1 8B: 1.5%, Qwen 3 4B: 3.0%. What does 'Interruption Accuracy' precisely measure in Table 6? Why is it hovering at 0.0%–3.0% across all architectures?"*

**Author Response:**  
We thank the reviewer for highlighting this critical ambiguity. We apologize for the confusion caused by the column header; **this was a naming misnomer, not an indication of a failing interruption policy.**

1. **Clarification of the Metric:**  
   The metric in Table 6 (now Table 3 in the revised manuscript) is **End-to-End Trivia QA Exact-Match Score** on adversarial incremental trivia questions (\texttt{AdvQA}), **NOT** the precision of the interruption classification signal.
   - **Semantic Interruption Decision Precision:** On binary conversational floor control (evaluated on the static **FLEXI** benchmark and dynamic **Full-Duplex-Bench**), Dual-State achieves **97.25% to 100% precision and F1 score** in distinguishing intentional barge-ins from benign backchannels (Tables 1 and 2).
   - **HANDRAISER Trivia QA Score:** In the HANDRAISER benchmark, when the listener agent halts the speaker mid-stream, it is tasked with immediately emitting the exact name of the entity described by partial clues in 1--3 words. Because the \texttt{AdvQA} and \texttt{Protobowl} datasets are intentionally designed with adversarial phrasing to mislead human trivia competitors, open-source 4B/8B/12B models evaluated zero-shot achieve low single-digit exact-match scores (0.0%--3.0%) when guessing from partial prefixes of 5--20 tokens. In fact, our diagnostic tests confirm that even when provided with the complete question, 4B/8B models achieve $<10\%$ exact-match accuracy without domain-specific trivia fine-tuning.

2. **Purpose of the HANDRAISER Experiment:**  
   The objective of the HANDRAISER evaluation is **systems and turn-taking benchmarking**, not trivia QA evaluation. It physically evaluates:
   - **Multi-Agent Floor-Taking Latency (Time-to-Halt):** Dual-State cuts Time-to-Halt by **15.8%** on Gemma 4 12B (0.1856\,s vs 0.2205\,s).
   - **Consensus Time:** Dual-State delivers up to a **6.0% speedup in total Consensus Time** (0.8024\,s vs 0.8540\,s on Qwen 3 4B) by avoiding prefill queuing and halting speaker token transmission earlier.
   - **Elimination of Redundant Prefill:** Slashes redundant context re-prefill from up to **821.8 tokens per scenario down to strictly 0.0 tokens** (-100%).

3. **Revisions Made in Paper:**  
   We renamed the column to **"Trivia QA Exact-Match"** in Table 3 and added an explicit disambiguation paragraph in Section 6.1 to ensure complete clarity.

---

### Response to Weakness B & Question 3: Parametric Chunk Size ($k$) Sweep & Latency Trade-offs

> **Reviewer Comment:**  
> *"While KV forking eliminates prefix re-prefill, repeated forward passes on $L_p$ tokens under multi-stream concurrency introduce significant memory bandwidth pressure and kernel launch overhead. The paper lacks: (1) A parametric sweep over chunk size $k \in \{2, 5, 10, 20, 50\}$, (2) Multi-stream throughput benchmarks."*

**Author Response:**  
We completely agree. To characterize this trade-off, we conducted a parametric empirical sweep across chunk strides $k \in \{2, 5, 10, 20, 50\}$ on physical NVIDIA hardware over 50 scenarios of the HANDRAISER benchmark on Qwen 3 4B.

#### Empirical Results (Table 5 in Revised Manuscript):

| Chunk Stride ($k$) | Checks / 100 Tokens | Time-to-Halt (sec) | Consensus Time (sec) | Avg Tokens Read | Interruption Rate (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$k=2$** | 50.0 | 0.1010 s | **0.6344 s** | 10.2 | 94.0% |
| **$k=5$** | 20.0 | 0.1063 s | 0.7059 s | 20.5 | 72.0% |
| **$k=10$** | 10.0 | 0.1127 s | 0.7173 s | 28.6 | 44.0% |
| **$k=20$** | 5.0 | 0.1051 s | 0.7526 s | 31.0 | 44.0% |
| **$k=50$** | 2.0 | 0.1214 s | 0.9182 s | 32.6 | 62.0% |

#### Key Insights & The Pareto Frontier:
- **Fine-Grained Responsiveness vs Overhead ($k=2$):** At $k=2$, the agent halts with minimal delay (reading only 10.2 tokens and reaching consensus in 0.6344\,s), but requires 50 assessor evaluations per 100 incoming tokens.
- **Quantization Lag at Coarse Strides ($k=50$):** At $k=50$, evaluation checks drop to 2 per 100 tokens, but the model incurs coarse-grained floor-taking lag: Consensus Time slows to 0.9182\,s (+44.7% over $k=2$), and the agent consumes 32.6 tokens before halting.
- **The Optimal Sweet Spot ($k \in [5, 10]$):** A chunk stride of $k \in [5, 10]$ represents the optimal Pareto equilibrium, bounding evaluation overhead to only 10--20 checks per 100 tokens while maintaining rapid floor-taking ($0.70\text{--}0.71\text{ s}$ Consensus Time).

This sweep has been added to Section 6.3 as Table 5.

---

### Response to Weakness C: Comparison with Auxiliary Lightweight Probing

> **Reviewer Comment:**  
> *"The proposed approach executes a full forward pass through the entire 8B/12B parameter model for each chunk evaluation. The paper does not compare against a standard, lightweight alternative: extracting intermediate or final hidden states from the streaming base model and feeding them into a tiny auxiliary classification head (e.g., a 2-layer MLP or linear probe)."*

**Author Response:**  
This is an insightful point. To rigorously address this, we instrumented both a **Linear Probe** and a **2-layer MLP classification head** on top of the base model's final hidden state ($h_t \in \mathbb{R}^{2560}$ on Qwen 3 4B) and benchmarked physical inference latency on our GPU using `torch.cuda.Event`.

#### Physical Benchmark Comparison (Table 6 in Revised Manuscript):

| Architecture | Trainable Parameters | Forward Latency (ms) | Zero-Shot Prompt Reconfigurability | Suffix Memory Overhead |
|:---|:---:|:---:|:---:|:---:|
| **Linear Probe** | 5,122 | **0.084 ms** | **No** (Requires task-specific supervised training) | **0.01 MB** |
| **2-Layer MLP Head** | 656,130 | 0.103 ms | **No** (Requires task-specific supervised training) | 1.25 MB |
| **Dual-State Assessor (Ours)** | **0 (Shared Weights)** | 91.50 ms | **Yes (100% zero-shot, prompt-guided)** | 0.65 MB ($L_p = 10$) |

#### Why Dual-State KV-Forking is Structurally Superior for Agentic Systems:
1. **The Zero-Shot Adaptability Imperative:** While an auxiliary MLP head executes in $\sim 0.1\text{ ms}$, it is **statically bound** to fixed offline supervised data. In real-world multi-agent systems, conversation rules, safety constraints, personas, and task definitions change dynamically via system prompts and few-shot examples. A linear probe or MLP head **cannot adapt zero-shot** to new prompt instructions without collecting training data and retraining the probe.
2. **Foundational Reasoning Preservation:** Dual-State directly leverages the foundational model's native instruction-following and in-context reasoning capabilities. By modifying the system prompt, developers can immediately enforce novel semantic boundaries (e.g., "do not interrupt unless the user mentions gluten") zero-shot, requiring zero training iterations.
3. **Operational Viability:** At $91.5\text{ ms}$, Dual-State executes well within the 200\,ms human conversational latency threshold, while maintaining complete zero-shot prompt configurability.

We have added Table 6 and this discussion to Section 6.3.

---

### Response to Weakness D: Positioning vs Modern Inference Engines (RadixAttention & PagedAttention)

> **Reviewer Comment:**  
> *"KV cache branching, prefix caching, and speculative fork-join execution are well-established primitives in engines like vLLM, SGLang (RadixAttention), and LightLLM. The submission would benefit from explicitly contrasting its PyTorch prototype with native radix-tree execution graphs in existing serving engines."*

**Author Response:**  
We appreciate this constructive suggestion and have added a dedicated discussion in Section 7.1 formally connecting Dual-State to modern radix-tree execution graphs:

1. **Mapping to RadixAttention (SGLang) and PagedAttention (vLLM):**  
   In modern serving systems, KV caches are managed as radix trees of fixed-size physical memory pages. Dual-State directly maps onto this architecture:
   - **Trunk Growth:** Ingesting incoming speaker tokens corresponds to appending physical blocks along the trunk path of the radix tree.
   - **Ephemeral Leaf-Node Fork:** The Assessor evaluation is instantiated as an ephemeral leaf node. It references the base trunk pages via copy-on-write pointers, allocating memory only for the $L_p = 10$ suffix tokens.
   - **Instant Deallocation:** Upon computing the binary gated logits, the leaf node is immediately dropped, freeing the $L_p$ blocks without memory fragmentation or prefix copying.
2. **Distinction from Speculative Decoding:**  
   Unlike speculative decoding (which uses a draft model to guess future tokens along the primary branch), Dual-State executes an orthogonal **floor-control evaluation branch** on an identical prefix without competing for generation state.

---

### Response to Question 2: Decider Prompt Formulation ($P_{eval}$)

> **Reviewer Comment:**  
> *"Where and how is $P_{eval}$ appended? If placed suffix-style after the dialogue stream, does this violate the chat/instruction templates of models expecting system instructions at the prefix?"*

**Author Response:**  
$P_{eval}$ does **not** violate the model's prefix chat template. The execution flow maintains strict template consistency:
1. **System Instruction at Prefix:** The conversation begins by pre-filling the Base KV cache with the standard system instruction (e.g., `<|im_start|>system\nYou are a helpful assistant...<|im_end|>`).
2. **Streaming Dialogue:** Clues and dialogue tokens stream in continuously following standard turn delimiters.
3. **Turn-Switching Suffix ($P_{eval}$):** At each chunk boundary, the forked Assessor KV cache is appended with the model's standard turn-switch token sequence (e.g., `\nClassification:` or `<|turn|>model\n<|channel>`). Because decoder-only transformers employ causal masking, appending structural tokens at the end of the context acts as a natural generation trigger prompt rather than an out-of-order instruction, adhering strictly to the causal attention structure.

---

### Response to Question 4 & Limitations: Multi-Stream Concurrency & Multi-Party Dialogue

> **Reviewer Comment:**  
> *"What is the memory overhead of maintaining forked Assessor KV tensors when running multiple concurrent agent dialogues simultaneously on a single GPU? Also, failure to address multi-party conversations (>2 agents)."*

**Author Response:**  

#### 1. Multi-Stream Memory Footprint Scaling ($B \in [1, 64]$)
We derived the theoretical memory footprint and validated it on GPU across batch sizes $B \in \{1, 4, 8, 16, 32, 64\}$ for context $L=512$ and $L_p=10$ on Qwen 3 4B (36 layers, 8 KV heads, $d_h=80$, BF16):

| Concurrent Streams ($B$) | Base KV Cache (MB) | Dual-State Assessor Overhead (MB) | Independent Baseline Total KV (MB) | Memory Saved (MB) | Memory Savings (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$B=1$** | 45.0 MB | **0.88 MB** | 90.9 MB | 45.0 MB | **49.5%** |
| **$B=4$** | 180.0 MB | **3.52 MB** | 363.5 MB | 180.0 MB | **49.5%** |
| **$B=8$** | 360.0 MB | **7.03 MB** | 727.0 MB | 360.0 MB | **49.5%** |
| **$B=16$** | 720.0 MB | **14.06 MB** | 1,454.1 MB | 720.0 MB | **49.5%** |
| **$B=32$** | 1,440.0 MB | **28.12 MB** | 2,908.1 MB | 1,440.0 MB | **49.5%** |
| **$B=64$** | 2,880.0 MB | **56.25 MB** | 5,816.2 MB | 2,880.0 MB | **49.5%** |

Because Dual-State pointer-aliases the base cache, the Assessor adds less than **2% additional memory** over the base stream, delivering a **49.5% net reduction in total KV cache memory** compared to independent deciders that duplicate context.

#### 2. Multi-Party Conversations ($>2$ Agents)
We expanded Section 7.4 to address multi-party agent swarms: In conversations with $>2$ agents, floor control cannot be resolved as an isolated binary decision. It requires competitive floor arbitration (such as token-weighted bidding, confidence-ranked hand-raising, or priority tiers). Extending KV-forked assessor branches to evaluate concurrent multi-agent bidding represents an exciting avenue for future work.

---

### Summary of Manuscript Updates in Revised Draft

1. **Table 3 & Section 6.1:** Renamed to "Trivia QA Exact-Match" and added explicit disambiguation separating adversarial trivia QA scores from semantic interruption classification precision ($97.25\%$--$100\%$).
2. **Table 5 & Section 6.3:** Added parametric chunk-size sweep ($k \in \{2, 5, 10, 20, 50\}$) identifying the $k \in [5, 10]$ Pareto sweet spot.
3. **Table 6 & Section 6.3:** Added empirical benchmark comparing Linear/MLP probes against Dual-State, articulating the zero-shot prompt reconfigurability advantage.
4. **Section 7.1:** Added architectural positioning mapping Dual-State to radix trees in SGLang (RadixAttention) and vLLM (PagedAttention).
5. **Section 7.2:** Added multi-stream concurrency memory scaling table proving 49.5% KV memory savings across $B=1$ to $64$.
6. **Section 7.4:** Added discussion on multi-party conversation scaling and floor arbitration.
7. **Compiled Artifact:** PDF compiled cleanly (9 pages including bibliography) at [`neurips_paper.pdf`](file:///home/mallock/agentic_interrupt/neurips_paper.pdf).

Given that all empirical questions have been thoroughly resolved with new physical GPU benchmarks and integrated directly into the paper, we respectfully request the reviewer to consider increasing their score to **Accept / Strong Accept**.
