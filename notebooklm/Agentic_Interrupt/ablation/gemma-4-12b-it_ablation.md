### Evaluation Table: Gemma 4 12B Architecture Ablation

| [Gemma 4 12B] | Independent Decider (Baseline) | Independent Decider (Thought Bypass) | Independent Decider (Logit Gating) | Dual LLM (Logit Gating) |
|---|---|---|---|---|
| **FLEXI Accuracy** | **99.75%**<br>*(TP: 200, TN: 199, FP: 1, FN: 0)* | **99.75%**<br>*(TP: 200, TN: 199, FP: 1, FN: 0)* | **98.75%** | **98.75%**<br>*(TP: 200, TN: 195, FP: 5, FN: 0)* |
| **FDB Accuracy** | **99.50%**<br>*(TP: 198, TN: 200, FP: 0, FN: 2)* | **100.00%**<br>*(TP: 200, TN: 200, FP: 0, FN: 0)* | **90.75%** | **100.00%**<br>*(TP: 200, TN: 200, FP: 0, FN: 0)* |
| **Avg FLEXI Latency** | 1063.27 ms | 823.35 ms | 260.77 ms | 95.23 ms (Blended) |
| **Avg FDB Latency** | 2452.00 ms | 1047.49 ms | 266.06 ms | 90.26 ms (Blended) |
| **Interruption Latency (Average latency of TP questions)** | FLEXI: 1131.24 ms<br>FDB: 1117.61 ms | FLEXI: 759.33 ms<br>FDB: 746.42 ms | FLEXI: 265.01 ms<br>FDB: 268.46 ms | FLEXI: 190.46 ms<br>FDB: 180.53 ms |
| **No-Op Latency Impact (Average latency of TN questions)** | FLEXI: 996.35 ms<br>FDB: 3572.25 ms | FLEXI: 888.20 ms<br>FDB: 1348.56 ms | FLEXI: 256.39 ms<br>FDB: 262.25 ms | **~2.00 ms (Thread Overhead)** (Parallel) |
