import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
with open("quizbowl_data/full.json") as f:
    data = json.load(f)

print(f"Total scenarios in HANDRAISER benchmark: {len(data)}")

chunk_size = 5
assessor_prompt_len = 10 # Structural instruction prompt tokens
decider_output_len_baseline = 15 # Baseline emits ~15 reasoning tokens

total_chunks = 0
total_stream_tokens = 0
baseline_prefill_tokens_total = 0
dual_prefill_tokens_total = 0
baseline_output_tokens_total = 0
dual_assessor_prompt_tokens_total = 0
dual_assessor_output_tokens_total = 0

per_scenario_stats = []

for item in data:
    text = item["question"]
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    seq_len = len(token_ids)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    total_chunks += num_chunks
    total_stream_tokens += seq_len
    
    # Baseline: at chunk i (1 to num_chunks), prefills i * chunk_size context tokens + prompt
    # plus emits ~15 decider tokens each check
    baseline_prefill_tokens = sum(min(i * chunk_size, seq_len) for i in range(1, num_chunks + 1))
    baseline_output_tokens = num_chunks * decider_output_len_baseline
    
    # Dual-state:
    # Base stream prefill: seq_len (each token ingested exactly once)
    # Redundant context prefill: 0
    # Assessor prompt tokens: num_chunks * assessor_prompt_len
    # Assessor output tokens: num_chunks * 1 (logit gated)
    dual_context_reprefill = 0
    dual_assessor_prompts = num_chunks * assessor_prompt_len
    dual_assessor_outputs = num_chunks * 1
    
    baseline_prefill_tokens_total += baseline_prefill_tokens
    baseline_output_tokens_total += baseline_output_tokens
    dual_assessor_prompt_tokens_total += dual_assessor_prompts
    dual_assessor_output_tokens_total += dual_assessor_outputs
    
    per_scenario_stats.append({
        "seq_len": seq_len,
        "num_chunks": num_chunks,
        "baseline_prefill": baseline_prefill_tokens,
        "dual_reprefill": 0,
        "dual_assessor_prompts": dual_assessor_prompts,
        "dual_assessor_outputs": dual_assessor_outputs
    })

avg_chunks = total_chunks / len(data)
avg_stream_len = total_stream_tokens / len(data)
avg_baseline_prefill = baseline_prefill_tokens_total / len(data)
avg_baseline_output = baseline_output_tokens_total / len(data)
avg_dual_prompts = dual_assessor_prompt_tokens_total / len(data)
avg_dual_outputs = dual_assessor_output_tokens_total / len(data)

print(f"\n--- TOKEN ACCOUNTING RESULTS (HANDRAISER Benchmark, 200 Scenarios) ---")
print(f"Average Stream Length: {avg_stream_len:.1f} tokens ({avg_chunks:.1f} chunks of k={chunk_size})")
print(f"\n1. Redundant Context Re-Prefill Tokens:")
print(f"   - Baseline (Independent Decider): {avg_baseline_prefill:.1f} tokens per scenario (Total across 200: {baseline_prefill_tokens_total:,})")
print(f"   - Dual-State KV-Forking:          0.0 tokens (Total: 0)")
print(f"\n2. Assessor Evaluation Tokens (The actual compute spent by Assessor):")
print(f"   - Assessor Prompt Suffix Tokens:  {avg_dual_prompts:.1f} tokens per scenario ({assessor_prompt_len} per chunk)")
print(f"   - Assessor Output Decider Tokens: {avg_dual_outputs:.1f} tokens per scenario (1 per chunk)")
print(f"   - Total Assessor Tokens:          {avg_dual_prompts + avg_dual_outputs:.1f} tokens per scenario")
print(f"\n3. Grand Total Tokens Processed per Scenario (Stream + Checks + Outputs):")
print(f"   - Independent Baseline: {avg_stream_len + avg_baseline_prefill + avg_baseline_output:.1f} tokens")
print(f"   - Dual-State Proposed:  {avg_stream_len + avg_dual_prompts + avg_dual_outputs:.1f} tokens")
print(f"   - Net Token Reduction:  {((avg_stream_len + avg_baseline_prefill + avg_baseline_output) - (avg_stream_len + avg_dual_prompts + avg_dual_outputs)) / (avg_stream_len + avg_baseline_prefill + avg_baseline_output) * 100:.1f}% reduction")

summary_data = {
    "avg_stream_len": avg_stream_len,
    "avg_chunks": avg_chunks,
    "avg_baseline_prefill": avg_baseline_prefill,
    "avg_dual_context_reprefill": 0.0,
    "avg_dual_assessor_prompts": avg_dual_prompts,
    "avg_dual_assessor_outputs": avg_dual_outputs,
    "total_baseline_tokens": avg_stream_len + avg_baseline_prefill + avg_baseline_output,
    "total_dual_tokens": avg_stream_len + avg_dual_prompts + avg_dual_outputs
}
with open("token_accounting_summary.json", "w") as f:
    json.dump(summary_data, f, indent=2)
