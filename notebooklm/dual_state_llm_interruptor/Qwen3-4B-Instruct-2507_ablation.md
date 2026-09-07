### Evaluation Table: Qwen3 4B Instruct Architecture Ablation

| [Qwen3 4B Instruct] | Independent Decider (Baseline) | Independent Decider (Logit Gating) | Dual LLM (Logit Gating) |
|---|---|---|---|
| **FLEXI Accuracy** | **97.50%**<br>*(TP: 198, TN: 192, FP: 8, FN: 2)* | **98.75%**<br>*(TP: 198, TN: 197, FP: 3, FN: 2)* | **99.00%**<br>*(TP: 198, TN: 198, FP: 2, FN: 2)* |
| **FDB Accuracy** | **99.75%**<br>*(TP: 199, TN: 200, FP: 0, FN: 1)* | **99.75%**<br>*(TP: 199, TN: 200, FP: 0, FN: 1)* | **99.50%**<br>*(TP: 198, TN: 200, FP: 0, FN: 2)* |
| **Avg FLEXI Latency** | 221.54 ms | 124.52 ms | 133.94 ms |
| **Avg FDB Latency** | 189.64 ms | 122.35 ms | 138.45 ms |
| **Interruption Latency (Average latency of TP questions)** | FLEXI: 261.13 ms<br>FDB: 191.19 ms | FLEXI: 123.57 ms<br>FDB: 122.63 ms | FLEXI: 133.94 ms<br>FDB: 138.45 ms |
| **No-Op Latency Impact (Average latency of TN questions)** | FLEXI: 181.73 ms<br>FDB: 188.16 ms | FLEXI: 126.17 ms<br>FDB: 121.97 ms | **~2.00 ms (Thread Overhead)** (Parallel) |
