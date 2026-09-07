### Evaluation Table: Gemma 4 12B Architecture Ablation

| [FOR MODEL GEMMA 4 12B] | Independent Decider Model (Baseline) | Decider Model + Prefilled Thinking (1-Token Hack) | Decider model + logit Gating (1-Token Native) | Dual LLM State + logit gating (Proposed) |
|-------------------------|--------------------------------------|---------------------------------------------------|-----------------------------------------------|------------------------------------------|
| **Benchmark1 Latency** (FLEXI TP) | **11,735.14 ms** <br> *(6,665ms Prefill + 5,069ms Gen)* | **6,665.34 ms** <br> *(6,665ms Prefill + 1-token)* | **6,665.34 ms** <br> *(6,665ms Prefill + 1-token)* | **401.46 ms** <br> *(0ms Prefill + 1-token)* |
| **Benchmark2 Latency** (FDB TP)   | **11,735.14 ms**                     | **6,665.34 ms**                             | **6,665.34 ms**                             | **417.53 ms**             |
| **No-Op dataset latency impact**| **6,665.34 ms per chunk**<br>*(Hard-Blocking prefill penalty)* | **6,665.34 ms per chunk**<br>*(Hard-Blocking prefill penalty)* | **6,665.34 ms per chunk**<br>*(Hard-Blocking prefill penalty)* | **401.46 ms average**<br>*(Blocking: 0 ms. Runs parallel via O(1) memory pointer)* |
| **Benchmark1 Recall** (FLEXI TP) | **98.5%** *(Full Reasoning Context)* | **92.25%** *(Via Boundary Few-Shot)*              | **97.25%** *(Logit Masking Constraint)*       | **97.25%** *(Logit Masking Constraint)*    |
| **Benchmark2 Precision** (FDB FP)| ~98.0% *(Contextually Aware)*        | **100.0%** *(Highly constrained intent)*          | **100.0%** *(Constraint Decoding)*            | **100.0%** *(Constraint Decoding)*         |
