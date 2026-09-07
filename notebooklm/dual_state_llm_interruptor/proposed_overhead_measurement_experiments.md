# Proposed Overhead Measurement Experiments
**Redefining Latency and Compute Benchmarks for Asynchronous Semantic VAD**

## The Asynchronous Paradigm Shift
In previous evaluations, the "No-Op Latency Impact" was measured under the assumption that an Independent Decider architecture would hard-block the main generation thread. However, a properly engineered system can launch any Independent Decider asynchronously. In an async paradigm, the main generation thread never pauses during a benign backchannel—rendering the perceived No-Op latency effectively `~0 ms` across all architectures.

While asynchronous execution masks the blocking penalty, it exposes the true underlying bottlenecks of the Independent Decider approach: **GPU Contention, Memory Duplication, and Late Halting**. 

To accurately capture the physical differences between an Async Independent Decider and an Async Dual-State LLM, we propose a new 4-metric Compute & Degradation Benchmark.

---

## The 4-Metric Benchmarking Matrix

### 1. Time-to-Halt (Interruption Latency)
**Definition**: The wall-clock time from the exact moment the user injects a True Positive (TP) barge-in, to the moment the system processes the audio, evaluates the semantics, and actually halts the Base LLM.
**The Problem**: If an async Independent Decider requires 3,500 ms to compute a massive prefill before outputting `STOP`, the bot will awkwardly talk over the user for 3.5 seconds before halting. 
**Measurement**: 
- **Async Independent Decider**: Slower Time-to-Halt (scales with context length `N`).
- **Async Dual-State**: Sub-200ms real-time Time-to-Halt (constant `O(1)` scaling).

### 2. Main-Stream TPS Degradation (GPU Contention)
**Definition**: The percentage drop in the Base LLM's Tokens-Per-Second (TPS) generation speed while a background True Negative (TN) backchannel evaluation is actively occurring.
**The Problem**: Launching a massive prefill in the background competes directly with the Base LLM for CUDA cores and memory bandwidth. 
**Measurement**:
- Run a continuous generation loop on the Base LLM.
- Inject asynchronous `IGNORE`/`CONTINUE` evaluations in the background.
- Measure the dip in TPS. The Dual-State architecture should demonstrate near-zero TPS degradation because it avoids the computationally heavy prefill phase.

### 3. Redundant Context Prefill (Compute Waste / FLOPs)
**Definition**: A mathematical measurement of the extra context tokens processed per VAD evaluation.
**The Problem**: An Independent Decider must re-process the entire conversation history `N` every single time the user makes a sound. If a user utters 10 backchannels in a long conversation, the system pays the compute cost of the full conversation 10 redundant times.
**Measurement**:
- **Async Independent Decider**: `N` redundant tokens processed per evaluation.
- **Async Dual-State**: `0` redundant tokens processed (only processes the new user audio chunk + structural prompt).

### 4. Peak VRAM Overhead (Memory Duplication)
**Definition**: The maximum extra GPU memory allocated strictly to evaluate the semantic VAD.
**The Problem**: An Independent Decider requires allocating a completely separate KV cache for the full conversation history. 
**Measurement**:
- Track peak CUDA memory allocated during the evaluation chunk.
- **Async Independent Decider**: `2x` memory overhead (full context duplication).
- **Async Dual-State**: `1x` memory overhead (leveraging zero-copy memory pointers for the shared history, plus a microscopic delta for the new branching tokens).

---

## Conclusion
By shifting the benchmark away from simple thread-blocking and focusing on **Degradation, Memory, and True Halt Time**, we establish a rigorous framework that scientifically proves the Dual-State architecture is computationally superior in an asynchronous environment.
