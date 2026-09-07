import time
import json
import threading
import queue
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

models = [
    ("Qwen3-4B", "Qwen/Qwen3-4B-Instruct-2507"),
    ("Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B-Instruct"),
    ("Gemma-4-12B", "unsloth/gemma-4-12b-it")
]

with open("quizbowl_data/full.json") as f:
    dataset = json.load(f) # All 200 scenarios

chunk_size = 5
token_delay_s = 0.015 # 15ms per token = 75ms per 5-token chunk

full_duplex_results = {}

prompt_template = """You are playing a game of progressive trivia. You receive clues about an entity token by token.
Current Clues: {context}

Question: Do you have enough high-confidence clues to definitively name the entity right now?
Answer EXACTLY 'STOP' or 'CONTINUE'."""

for model_name, model_id in models:
    print(f"\n==========================================")
    print(f"Running True Full-Duplex HANDRAISER for {model_name}...")
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
    
    stop_id = tokenizer.encode("STOP", add_special_tokens=False)[-1]
    continue_id = tokenizer.encode("CONTINUE", add_special_tokens=False)[-1]
    
    model_stats = {
        "indep": {"halt_times": [], "consensus_times": [], "tokens_read": [], "correct": 0, "passive": 0},
        "dual":  {"halt_times": [], "consensus_times": [], "tokens_read": [], "correct": 0, "passive": 0}
    }
    
    def run_duplex_scenario(item, mode="dual"):
        question_tokens = tokenizer.encode(item['question'], add_special_tokens=False)
        stream_queue = queue.Queue()
        stop_event = threading.Event()
        
        # Speaker thread streaming tokens
        def speaker_worker():
            for i in range(0, len(question_tokens), chunk_size):
                if stop_event.is_set():
                    break
                chunk = question_tokens[i:i+chunk_size]
                stream_queue.put(chunk)
                time.sleep(token_delay_s * len(chunk))
            stream_queue.put(None)
            
        speaker_thread = threading.Thread(target=speaker_worker)
        start_time = time.time()
        speaker_thread.start()
        
        interrupted = False
        context_tokens = []
        current_kv_cache = None
        halt_timestamp = None
        eval_time_last = 0.0
        
        while True:
            chunk = stream_queue.get()
            if chunk is None:
                break
            context_tokens.extend(chunk)
            
            if mode == "dual":
                inputs = torch.tensor([chunk]).to("cuda")
                with torch.no_grad():
                    base_out = model(inputs, past_key_values=current_kv_cache, use_cache=True)
                    current_kv_cache = base_out.past_key_values
                    
                assessor_inputs = tokenizer("\nClassification: ", return_tensors="pt").input_ids.to("cuda")
                t0 = time.time()
                with torch.no_grad():
                    assessor_out = model(assessor_inputs, past_key_values=current_kv_cache, use_cache=True)
                    logits = assessor_out.logits[:, -1, :]
                    s_logit = logits[0, stop_id].item()
                    c_logit = logits[0, continue_id].item()
                eval_time_last = time.time() - t0
                
                if s_logit > c_logit:
                    stop_event.set()
                    halt_timestamp = time.time()
                    interrupted = True
                    break
            else:
                context_text = tokenizer.decode(context_tokens)
                eval_prompt = prompt_template.format(context=context_text)
                inputs = tokenizer(eval_prompt, return_tensors="pt").input_ids.to("cuda")
                t0 = time.time()
                with torch.no_grad():
                    out = model(inputs, use_cache=True)
                    logits = out.logits[:, -1, :]
                    s_logit = logits[0, stop_id].item()
                    c_logit = logits[0, continue_id].item()
                eval_time_last = time.time() - t0
                
                if s_logit > c_logit:
                    stop_event.set()
                    halt_timestamp = time.time()
                    interrupted = True
                    break
                    
        speaker_thread.join()
        
        # Listener guess
        context_text = tokenizer.decode(context_tokens)
        if interrupted:
            guess_prompt = f"Description: {context_text}\nQuestion: Based on the description, what is the exact entity being described? (1-3 words only)\nAnswer: "
            guess_inputs = tokenizer(guess_prompt, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                gen_out = model.generate(guess_inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
            guess = tokenizer.decode(gen_out[0][len(guess_inputs[0]):], skip_special_tokens=True).strip()
        else:
            guess = "NONE"
            
        total_time = time.time() - start_time
        time_to_halt = eval_time_last
        
        # Simple accuracy check against target answer
        actual = str(item['answer']).lower()
        pred = guess.lower()
        is_correct = (actual in pred) or (pred in actual) if len(pred) > 1 else False
        
        return {
            "interrupted": interrupted,
            "tokens_read": len(context_tokens),
            "time_to_halt": time_to_halt,
            "consensus_time": total_time,
            "is_correct": is_correct
        }

    for idx, item in enumerate(dataset):
        res_indep = run_duplex_scenario(item, mode="indep")
        res_dual = run_duplex_scenario(item, mode="dual")
        
        model_stats["indep"]["halt_times"].append(res_indep["time_to_halt"])
        model_stats["indep"]["consensus_times"].append(res_indep["consensus_time"])
        model_stats["indep"]["tokens_read"].append(res_indep["tokens_read"])
        if res_indep["is_correct"]: model_stats["indep"]["correct"] += 1
        if not res_indep["interrupted"]: model_stats["indep"]["passive"] += 1
        
        model_stats["dual"]["halt_times"].append(res_dual["time_to_halt"])
        model_stats["dual"]["consensus_times"].append(res_dual["consensus_time"])
        model_stats["dual"]["tokens_read"].append(res_dual["tokens_read"])
        if res_dual["is_correct"]: model_stats["dual"]["correct"] += 1
        if not res_dual["interrupted"]: model_stats["dual"]["passive"] += 1
        
        if (idx + 1) % 25 == 0:
            print(f"Progress: {idx+1}/200 scenarios | Indep Avg Time: {sum(model_stats['indep']['consensus_times'])/len(model_stats['indep']['consensus_times']):.3f}s | Dual Avg Time: {sum(model_stats['dual']['consensus_times'])/len(model_stats['dual']['consensus_times']):.3f}s")
            
    # Aggregate
    full_duplex_results[model_name] = {
        "indep": {
            "avg_time_to_halt_s": round(sum(model_stats["indep"]["halt_times"])/200, 4),
            "avg_consensus_time_s": round(sum(model_stats["indep"]["consensus_times"])/200, 4),
            "accuracy": round(model_stats["indep"]["correct"] / 200 * 100, 2),
            "passive_rate": round(model_stats["indep"]["passive"] / 200 * 100, 2),
            "avg_tokens_read": round(sum(model_stats["indep"]["tokens_read"])/200, 1)
        },
        "dual": {
            "avg_time_to_halt_s": round(sum(model_stats["dual"]["halt_times"])/200, 4),
            "avg_consensus_time_s": round(sum(model_stats["dual"]["consensus_times"])/200, 4),
            "accuracy": round(model_stats["dual"]["correct"] / 200 * 100, 2),
            "passive_rate": round(model_stats["dual"]["passive"] / 200 * 100, 2),
            "avg_tokens_read": round(sum(model_stats["dual"]["tokens_read"])/200, 1)
        }
    }
    
    ind_c = full_duplex_results[model_name]["indep"]["avg_consensus_time_s"]
    dua_c = full_duplex_results[model_name]["dual"]["avg_consensus_time_s"]
    speedup = (ind_c - dua_c) / ind_c * 100
    print(f"\n{model_name} Full-Duplex Results:")
    print(f"  Indep Consensus Time: {ind_c:.4f}s | Dual Consensus Time: {dua_c:.4f}s ({speedup:+.1f}% speedup!)")
    print(f"  Indep Time-to-Halt:   {full_duplex_results[model_name]['indep']['avg_time_to_halt_s']:.4f}s | Dual Time-to-Halt: {full_duplex_results[model_name]['dual']['avg_time_to_halt_s']:.4f}s")
    print(f"  Indep Accuracy:       {full_duplex_results[model_name]['indep']['accuracy']}% | Dual Accuracy: {full_duplex_results[model_name]['dual']['accuracy']}%")
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

with open("true_full_duplex_benchmark_results.json", "w") as f:
    json.dump(full_duplex_results, f, indent=2)

print("\nAll Full-Duplex evaluations completed and saved to true_full_duplex_benchmark_results.json!")
