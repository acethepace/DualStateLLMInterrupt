### Evaluation Table: Llama 3.1 8B Instruct Architecture Ablation

| [Llama 3.1 8B] | Independent Decider (Baseline) | Independent Decider (Logit Gating) | Dual LLM (Logit Gating) |
|---|---|---|---|
| **FLEXI Accuracy** | **93.50%**<br>*(TP: 177, TN: 197, FP: 3, FN: 23)* | **91.00%**<br>*(TP: 198, TN: 197, FP: 3, FN: 2)* | **90.50%**<br>*(TP: 162, TN: 200, FP: 0, FN: 38)* |
| **FDB Accuracy** | **97.00%**<br>*(TP: 188, TN: 200, FP: 0, FN: 12)* | **95.00%**<br>*(TP: 199, TN: 200, FP: 0, FN: 1)* | **94.25%**<br>*(TP: 177, TN: 200, FP: 0, FN: 23)* |
| **Avg FLEXI Latency** | 384.77 ms | 117.60 ms | 78.91 ms (approx) |
| **Avg FDB Latency** | 361.96 ms | 117.07 ms | 115.73 ms (approx) |
| **Interruption Latency (Average latency of TP questions)** | FLEXI: 555.19 ms<br>FDB: 518.45 ms | FLEXI: 119.79 ms<br>FDB: 117.67 ms | FLEXI: 78.91 ms<br>FDB: 115.73 ms |
| **No-Op Latency Impact (Average latency of TN questions)** | FLEXI: 215.64 ms<br>FDB: 202.22 ms | FLEXI: 116.05 ms<br>FDB: 116.49 ms | **~2.00 ms (Thread Overhead)** (Parallel) |
