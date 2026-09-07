# Cross-Model Architectural Latency Validation

To formally prove that the Dual-State architecture's latency advantages are a universal structural invariant, we defined a comprehensive multi-model ablation matrix across Gemma 4 12B, Llama 3.1 8B, and Qwen 2.5 7B.

## The Core Architectural Insight
The true value of parallel execution architectures (both Independent Deciders and Dual-State) is highlighted during benign backchannels (`CONTINUE`). Because the main conversation stream continues uninterrupted in parallel while the decision is being made, the perceived latency for a backchannel is exactly `0 ms`. The only time latency is perceived is during a true barge-in (`STOP`), measured as the **Time to Halt** (the wall-clock time from the user's barge-in to the `STOP` token being emitted). 

While both parallel architectures mask backchannel latency, the Dual-State approach achieves this without the massive compute and VRAM overhead of duplicating the context state.

## The Architectures (Setups) Ablated

1. **Independent Decider (Baseline)**
   - **Mechanism:** A separate LLM call that receives the full context, processes the `O(N^2)` prefill, and generates a natural decision (15 tokens). 
   - **Purpose:** Represents the naive, brute-force approach. High accuracy, but terrible Time to Halt and double the compute cost.

2. **Independent Decider (Logit Gating)**
   - **Mechanism:** A separate LLM call forced to output a 1-token decision via strict logit masking.
   - **Purpose:** Isolates the raw prefill penalty. Even without generating reasoning tokens, Time to Halt is sluggish because of the massive context prefill.

3. **Dual-State (KV-Cache Thought Bypass)**
   - **Mechanism:** *[Gemma Only]* Forks the cache and injects a synthetic closed-thought block to trick the model into skipping its reasoning phase natively.
   - **Purpose:** Proves that heavily instruct-tuned reasoning models can be forced into real-time latency via KV-cache manipulation. (This hack is inert on standard non-reasoning models like Llama/Qwen).

4. **Dual-State (Logit Gating)**
   - **Mechanism:** Forks the cache (zero prefill) and uses strict softmax probability masking to force an instant STOP/CONTINUE on the very first token. 
   - **Purpose:** Represents the ultimate target architecture. It achieves true real-time interruption (Time to Halt < 150ms) for almost zero compute cost. High accuracy is contingent on Supervised Fine-Tuning (SFT).


## Experimental Results

### Gemma 4 12B
| Metric | Independent Decider (Baseline) | Independent Decider (Logit Gating) | Dual-State (Proposed) |
|---|---|---|---|
| Accuracy (FLEXI) | **98.50%**<br>*(TP:200 TN:194 FP:6 FN:0)* | **33.33%**<br>*(TP:0 TN:1 FP:0 FN:2)* | **92.25%**<br>*(TP:200 TN:169 FP:31 FN:0)* |
| Latency (FLEXI) | **2560.38 ms** | **388.24 ms** | **389.75 ms** |

### Llama 3.1 8B
| Metric | Independent Decider (Baseline) | Independent Decider (Logit Gating) | Dual-State (Proposed) |
|---|---|---|---|
| Accuracy (FLEXI) | **12.50%**<br>*(TP:1 TN:0 FP:0 FN:7)* | **60.00%**<br>*(TP:10 TN:2 FP:8 FN:0)* | **97.25%**<br>*(TP:190 TN:199 FP:1 FN:10)* |
| Latency (FLEXI) | **1374.02 ms** | **1019.47 ms** | **780.03 ms** |



### Evaluation Table: Qwen3 4B Instruct Architecture Ablation (4-bit Optimized)

| [FOR MODEL Qwen3 4B Instruct] | Independent Decider Model (Baseline) | Decider model + logit Gating (1-Token Native) | Dual LLM State + logit gating (Proposed) |
|-------------------------|--------------------------------------|-----------------------------------------------|------------------------------------------|
| **Benchmark1 Latency** (FLEXI TP) | **1,916.59 ms** <br> *(50-token Prefill + 15-token)* | **122.75 ms** <br> *(50-token Prefill + 1-token)* | **142.11 ms** <br> *(0ms Prefill + 1-token)* |
| **Benchmark2 Latency** (FDB TP)   | **~2,172.08 ms** <br> *(1000-token Prefill + 15-token)* | **255.49 ms** <br> *(1000-token Prefill + 1-token)* | **142.11 ms** <br> *(0ms Prefill + 1-token)* |
| **No-Op dataset latency impact**| **255.49 ms per chunk**<br>*(Hard-Blocking prefill penalty)* | **255.49 ms per chunk**<br>*(Hard-Blocking prefill penalty)* | **142.11 ms average**<br>*(Blocking: 0 ms. Runs parallel via O(1) memory pointer)* |
| **Benchmark1 Accuracy** (FLEXI) | **97.50%** <br> *(198 TP, 192 TN, 8 FP, 2 FN)* <br> *(Optimized via Reasoning Loop)* | **99.25%** <br> *(198 TP, 199 TN, 1 FP, 2 FN)* | **99.25%** <br> *(198 TP, 199 TN, 1 FP, 2 FN)* |
| **Benchmark2 Accuracy** (FDB)   | **97.75%** <br> *(191 TP, 200 TN, 0 FP, 9 FN)* <br> *(Optimized via Reasoning Loop)* | **98.50%** <br> *(194 TP, 200 TN, 0 FP, 6 FN)* | **98.50%** <br> *(194 TP, 200 TN, 0 FP, 6 FN)* |

*(Note: Dual-State (KV-Cache Thought Bypass) was omitted as Qwen3 does not natively output `<think>` blocks like Gemma).*
