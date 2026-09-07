# Proposed HANDRAISER Dual-State Experiment
**Proving the Viability of Continuous Asynchronous Interruption in Multi-Agent Systems**

## Executive Summary
The HANDRAISER protocol demonstrated that multi-agent systems save significant communication costs when listening agents actively halt speaking agents upon reaching task-confidence. However, standard listener evaluations require continuous `O(N^2)` Independent Decider inference loops, which bankrupt the compute budget and cause massive latency. 

This experiment will mathematically prove that by substituting the Independent Decider with our **Dual-State KV-Forking architecture**, a listening agent can continuously evaluate its confidence at tiny 5-token intervals with near-zero latency, low redundant compute waste, and superior task accuracy.

---

## Experimental Setup

### 1. The Environment (Quizbowl / Trivia Datasets)
Instead of relying on a live LLM to generate Pictionary clues (which introduces variability), we will curate a deterministic, fixed-speaker dataset using the structure of Quizbowl. Quizbowl questions are explicitly designed to start with obscure clues and progressively become more obvious, making them the perfect proxy for testing interruption timing.

**Dataset Curation**: 
We will curate our evaluation dataset by sampling from:
* `qanta-challenge/AdvQA` (Adversarial human-crafted QA trivia designed to challenge models).
* `mgor/protobowl-11-13` (Historical Quizbowl logs containing progressive clues).

**The Streaming Mechanism**: 
* The Speaker agent is replaced by a deterministic script that streams the text of the Quizbowl question to the Listener.
* The text will be chunked and delivered exactly **5 tokens at a time**.

### 2. The Listener Architectures (Ablation)
Upon receiving each 5-token chunk, the Listener must evaluate if it has enough information to interrupt the Speaker and solve the task. We will test two architectures:
1. **Baseline (Independent Decider)**: The Listener spawns an asynchronous prompt reading the entire accumulated context to decide `STOP` or `CONTINUE`.
2. **The Invention (Dual-State LLM)**: The Listener natively ingests the 5 tokens into its Base State KV cache, and asynchronously forks that cache to an Assessor which outputs `STOP` or `CONTINUE` using Logit Gating.

---

## LLM-as-a-Judge: Interruption Accuracy & Task Success
A fast interruption is worthless if it leads to a hallucinated answer. We will utilize an advanced external judge (e.g., OpenAI's GPT-4o) to evaluate the **Interruption Accuracy**.

**The Evaluation Flow**:
1. The Listener triggers an interruption (`STOP`).
2. The Speaker halts.
3. The Listener executes its final action (e.g., outputs its Pictionary guess, or finalizes the meeting time).
4. **The Judge** evaluates the Listener's final answer. 
   - If the answer is correct/valid, the interruption was an **Accurate Interruption**.
   - If the answer is wrong/hallucinated, the interruption was **Premature (False Positive)**.

---

## The Metrics Matrix
To comprehensively prove the superiority of the Dual-State architecture in the HANDRAISER framework, we will track the following holistic metrics:

### A. Task & Accuracy Metrics
1. **Interruption Accuracy (Judge Score)**: Percentage of interruptions that led to a correct final answer.
2. **Task Success Rate**: Overall percentage of tasks successfully completed.
3. **Passive Listening Rate**: The percentage of times the Listener failed to interrupt at all, waiting for the Speaker to naturally hit its `<EOS>` token. (Lower is better; proves the Listener successfully seized the floor).

### B. Computational Overhead Metrics (Evaluated per 5-token chunk)
4. **Redundant Context Prefill (Tokens)**: The number of redundant tokens processed by the Listener to evaluate the interruption. (Expected: `N` for Baseline, `0` for Dual-State).
5. **Time-to-Halt (Interruption Latency)**: The wall-clock time from the Listener acquiring the sufficient 5th token, to the network signal physically halting the Speaker.
6. **Total Time to Consensus**: The total end-to-end wall-clock time required to complete the multi-agent task. (Expected to drop massively in Dual-State because evaluation happens in parallel without stalling).
7. **Peak VRAM Overhead**: Max GPU memory required to host the Listener's VAD sub-routines.

---

## Conclusion
By evaluating at hyper-granular 5-token intervals, the Baseline architecture will buckle under the exponential `O(N^2)` compute weight of continuous re-prefilling, drastically spiking the "Total Time to Consensus" and causing massive latency delays in halting the speaker. 

Conversely, the Dual-State architecture will evaluate every 5 tokens natively via `O(1)` zero-copy forking, proving that near-continuous semantic monitoring is not only possible, but highly efficient, accurate, and essential for state-of-the-art multi-agent protocols.
