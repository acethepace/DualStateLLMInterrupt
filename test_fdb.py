import sys
import os
import json
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from benchmark_runners.fdb_runner import simulate_fdb_multi_agent, get_templates
import benchmark_runners.fdb_runner as fdb
fdb.load_fdb_tasks = lambda x: json.load(open(x))["splits"]["test"]["tasks"][:3]
fdb.simulate_fdb_multi_agent("unsloth/gemma-4-12b-it", "v2_prefill", "datasets/Full-Duplex-Bench/v2/prompts_staged_200.json")
