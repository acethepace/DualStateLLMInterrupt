import sys
import os

from benchmark_runners import fdb_runner as fdb

if __name__ == "__main__":
    fdb.simulate_fdb_multi_agent("Qwen/Qwen3-4B-Instruct-2507", "v1", "datasets/Full-Duplex-Bench/v2/prompts_staged_200.json")
