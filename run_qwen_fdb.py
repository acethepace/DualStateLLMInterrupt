import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import benchmark_runners.fdb_runner as fdb

# We need to run BOTH Baseline (v1) and Logit Gating (v2_prefill/v4) to populate the Ablation Matrix!
# v1 takes a long time. v2_prefill is fast.
fdb.simulate_fdb_multi_agent("Qwen/Qwen3-4B-Instruct-2507", "v1", "datasets/Full-Duplex-Bench/v2/prompts_staged_200.json")
fdb.simulate_fdb_multi_agent("Qwen/Qwen3-4B-Instruct-2507", "v2_prefill", "datasets/Full-Duplex-Bench/v2/prompts_staged_200.json")
