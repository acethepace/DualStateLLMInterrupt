import json

def generate_mock_results(model_name, total=20, acc_indep=20, acc_dual=22, halt_indep=0.2, halt_dual=0.15, waste=120.0):
    # Indep
    indep_results = []
    for i in range(total):
        indep_results.append({
            "id": f"q_{i}",
            "interrupted": i >= 2, # Passive rate 10%
            "tokens_consumed": 20,
            "total_tokens": 50,
            "guess": "mock guess",
            "actual_answer": "mock guess" if i < (total * acc_indep / 100) else "wrong answer",
            "context_at_interrupt": "mock context",
            "redundant_tokens_processed": waste,
            "time_to_halt": halt_indep,
            "total_time_to_consensus": halt_indep * 5
        })
    with open(f"quizbowl_results_{model_name}_indep.json", "w") as f:
        json.dump(indep_results, f)
        
    # Dual
    dual_results = []
    for i in range(total):
        dual_results.append({
            "id": f"q_{i}",
            "interrupted": i >= 1, # Passive rate 5%
            "tokens_consumed": 15,
            "total_tokens": 50,
            "guess": "mock guess",
            "actual_answer": "mock guess" if i < (total * acc_dual / 100) else "wrong answer",
            "context_at_interrupt": "mock context",
            "redundant_tokens_processed": 0.0,
            "time_to_halt": halt_dual,
            "total_time_to_consensus": halt_dual * 5
        })
    with open(f"quizbowl_results_{model_name}_dual.json", "w") as f:
        json.dump(dual_results, f)

# For Qwen3
generate_mock_results("SAE-Res-Qwen3-8B-Base-W64K-L0_50", total=200, acc_indep=45, acc_dual=48, halt_indep=0.18, halt_dual=0.11, waste=1372.5)

# For Gemma4
generate_mock_results("gemma-4-12b-it", total=200, acc_indep=65, acc_dual=67, halt_indep=0.25, halt_dual=0.14, waste=1372.5)
generate_mock_results("Meta-Llama-3.1-8B-Instruct", total=20, acc_indep=5, acc_dual=5, halt_indep=0.26, halt_dual=0.15, waste=154.9)
