### Evaluation Table: Llama 3.1 8B Architecture Ablation (4-bit Optimized)

| [FOR MODEL Llama 3.1 8B] | Independent Decider Model (Baseline) | Decider Model + Prefilled Thinking (1-Token Hack) | Decider model + logit Gating (1-Token Native) | Dual LLM State + logit gating (Proposed) |
|-------------------------|--------------------------------------|---------------------------------------------------|-----------------------------------------------|------------------------------------------|
| **Benchmark1 Latency** (FLEXI TP) | **1,650.44 ms** <br> *(50-token Prefill + 15-token)* | **88.06 ms** <br> *(50-token Prefill + 1-token)* | **88.06 ms** <br> *(50-token Prefill + 1-token)* | **88.06 ms** <br> *(0ms Prefill + 1-token)* |
| **Benchmark2 Latency** (FDB TP)   | **~1,885.44 ms** <br> *(1000-token Prefill + 15-token)* | **323.47 ms** <br> *(1000-token Prefill + 1-token)* | **323.47 ms** <br> *(1000-token Prefill + 1-token)* | **153.66 ms** <br> *(0ms Prefill + 1-token)* |
| **No-Op dataset latency impact**| **323.47 ms per chunk**<br>*(Hard-Blocking prefill penalty)* | **323.47 ms per chunk**<br>*(Hard-Blocking prefill penalty)* | **323.47 ms per chunk**<br>*(Hard-Blocking prefill penalty)* | **153.66 ms average**<br>*(Blocking: 0 ms. Runs parallel via O(1) memory pointer)* |
| **Benchmark1 Recall** (FLEXI TP) | **10.0%** *(Zero-Shot)* | **8.0%** *(Zero-Shot)*              | **20.0%** *(Zero-Shot Logit Constraint)*       | **20.0%** *(Zero-Shot Logit Constraint)*    |
| **Benchmark2 Precision** (FDB FP)| **N/A** *(Failed Zero-Shot)*        | **N/A** *(Failed Zero-Shot)*          | **N/A** *(Failed Zero-Shot)*            | **N/A** *(Failed Zero-Shot)*         |
