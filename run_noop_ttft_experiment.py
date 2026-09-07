import time
import torch
import json
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

models = [
    ("Qwen3-4B", "Qwen/Qwen3-4B-Instruct-2507"),
    ("Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B-Instruct"),
    ("Gemma-4-12B", "unsloth/gemma-4-12b-it")
]

seq_lengths = [64, 128, 256, 512, 1024]
results = {}

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

for model_name, model_id in models:
    print(f"\n==========================================")
    print(f"Benchmarking No-Op TTFT & Latency for {model_name} (4-bit)...")
    print(f"==========================================")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="cuda",
        trust_remote_code=True
    )
    model.eval()
    
    results[model_name] = {
        "ttft_baseline_ms": {},
        "ttft_dual_streaming_ms": {},
        "kv_fork_time_ms": {},
        "assessor_forward_time_ms": {},
        "speedup_factor": {}
    }
    
    # Warmup
    dummy_input = tokenizer("Hello, how are you today?", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**dummy_input, use_cache=True)
        _ = model.generate(**dummy_input, max_new_tokens=2)
    torch.cuda.synchronize()
    
    # Generate context text
    base_text = "The quick brown fox jumps over the lazy dog and runs across the wide open grassy green meadow with great speed and agility. " * 80
    tokens_all = tokenizer(base_text, return_tensors="pt").input_ids[0].to("cuda")
    
    for seq_len in seq_lengths:
        context_tokens = tokens_all[:seq_len].unsqueeze(0)
        
        # 1. Baseline TTFT (Cold Prefill from scratch + 1 decode step)
        # Average over 3 runs for stability
        baseline_times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out_base = model(context_tokens, use_cache=True)
                next_token = torch.argmax(out_base.logits[:, -1, :], dim=-1, keepdim=True)
            torch.cuda.synchronize()
            baseline_times.append((time.perf_counter() - t0) * 1000.0)
        baseline_ttft = sum(baseline_times) / len(baseline_times)
        
        # 2. Dual-State / Input Streaming TTFT
        # Progressive chunks already computed and in KV cache.
        # Producing first response token from warmed KV cache only takes 1 decode step!
        past_kv = out_base.past_key_values
        dummy_query = next_token
        dual_times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out_stream = model(dummy_query, past_key_values=past_kv, use_cache=True)
                _ = torch.argmax(out_stream.logits[:, -1, :], dim=-1)
            torch.cuda.synchronize()
            dual_times.append((time.perf_counter() - t0) * 1000.0)
        dual_ttft = sum(dual_times) / len(dual_times)
        
        # 3. KV-Fork Overhead
        # Measure time to clone or prepare KV cache for assessor
        fork_times = []
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if hasattr(past_kv, "key_cache"): # DynamicCache in modern transformers
                forked_kv = []
                for k, v in zip(past_kv.key_cache, past_kv.value_cache):
                    forked_kv.append((k.clone(), v.clone()))
            elif isinstance(past_kv, tuple):
                forked_kv = tuple((k.clone(), v.clone()) for k, v in past_kv)
            else:
                forked_kv = past_kv
            torch.cuda.synchronize()
            fork_times.append((time.perf_counter() - t0) * 1000.0)
        kv_fork_time = sum(fork_times) / len(fork_times)
        
        # 4. Assessor Forward Time (1 token forward on forked cache)
        assessor_prompt_token = tokenizer.encode("\nAction:", return_tensors="pt").to("cuda")
        assessor_times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(assessor_prompt_token[:, :1], past_key_values=past_kv, use_cache=False)
            torch.cuda.synchronize()
            assessor_times.append((time.perf_counter() - t0) * 1000.0)
        assessor_forward_time = sum(assessor_times) / len(assessor_times)
        
        speedup = baseline_ttft / max(dual_ttft, 1e-4)
        
        results[model_name]["ttft_baseline_ms"][seq_len] = round(baseline_ttft, 2)
        results[model_name]["ttft_dual_streaming_ms"][seq_len] = round(dual_ttft, 2)
        results[model_name]["kv_fork_time_ms"][seq_len] = round(kv_fork_time, 2)
        results[model_name]["assessor_forward_time_ms"][seq_len] = round(assessor_forward_time, 2)
        results[model_name]["speedup_factor"][seq_len] = round(speedup, 2)
        
        print(f"L={seq_len:4d} tokens | Baseline TTFT: {baseline_ttft:6.2f} ms | Dual TTFT: {dual_ttft:5.2f} ms | Speedup: {speedup:5.1f}x | Fork: {kv_fork_time:4.2f} ms | Assessor Fwd: {assessor_forward_time:5.2f} ms")
        
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

with open("noop_ttft_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nBenchmark completed and saved to noop_ttft_benchmark_results.json")
