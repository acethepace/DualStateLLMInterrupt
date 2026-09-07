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
