### Evaluation Table: Qwen 2.5 7B Instruct Architecture Ablation (4-bit Optimized)

| [FOR MODEL Qwen 2.5 7B Instruct] | Independent Decider Model (Baseline) | Decider Model + Prefilled Thinking (1-Token Hack) | Decider model + logit Gating (1-Token Native) | Dual LLM State + logit gating (Proposed) |
|-------------------------|--------------------------------------|---------------------------------------------------|-----------------------------------------------|------------------------------------------|
| **Benchmark1 Latency** (FLEXI TP) | **1,447.36 ms** <br> *(50-token Prefill + 15-token)* | **65.58 ms** <br> *(50-token Prefill + 1-token)* | **65.58 ms** <br> *(50-token Prefill + 1-token)* | **94.87 ms** <br> *(0ms Prefill + 1-token)* |
| **Benchmark2 Latency** (FDB TP)   | **~1,737.35 ms** <br> *(1000-token Prefill + 15-token)* | **355.57 ms** <br> *(1000-token Prefill + 1-token)* | **355.57 ms** <br> *(1000-token Prefill + 1-token)* | **94.87 ms** <br> *(0ms Prefill + 1-token)* |
| **No-Op dataset latency impact**| **355.57 ms per chunk**<br>*(Hard-Blocking prefill penalty)* | **355.57 ms per chunk**<br>*(Hard-Blocking prefill penalty)* | **355.57 ms per chunk**<br>*(Hard-Blocking prefill penalty)* | **94.87 ms average**<br>*(Blocking: 0 ms. Runs parallel via O(1) memory pointer)* |
| **Benchmark1 Recall** (FLEXI TP) | **N/A** *(Zero-Shot omitted)* | **N/A** *(Zero-Shot omitted)*              | **N/A** *(Zero-Shot omitted)*       | **N/A** *(Zero-Shot omitted)*    |
| **Benchmark2 Precision** (FDB FP)| **N/A** *(Failed Zero-Shot)*        | **N/A** *(Failed Zero-Shot)*          | **N/A** *(Failed Zero-Shot)*            | **N/A** *(Failed Zero-Shot)*         |
