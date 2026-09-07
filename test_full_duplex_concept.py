import time
import json
import threading
import queue
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("Testing True Full-Duplex execution concept...")

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model_id = "unsloth/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="cuda")
model.eval()

stop_id = tokenizer.encode("STOP", add_special_tokens=False)[-1]
continue_id = tokenizer.encode("CONTINUE", add_special_tokens=False)[-1]

# Load 10 scenarios
with open("quizbowl_data/full.json") as f:
    scenarios = json.load(f)[:10]

chunk_size = 5
token_delay_s = 0.02 # 20ms per token = 100ms per 5-token chunk (realistic conversational pace)

print(f"Testing on {len(scenarios)} scenarios with simulated streaming pace ({token_delay_s*1000:.0f}ms/token)...")

def run_scenario(item, mode="dual"):
    question_tokens = tokenizer.encode(item['question'], add_special_tokens=False)
    
    stream_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Speaker Thread: streams tokens into queue at conversational pace
    speaker_tokens_emitted = []
    def speaker_worker():
        for i in range(0, len(question_tokens), chunk_size):
            if stop_event.is_set():
                break
            chunk = question_tokens[i:i+chunk_size]
            speaker_tokens_emitted.extend(chunk)
            stream_queue.put(chunk)
            time.sleep(token_delay_s * len(chunk))
        stream_queue.put(None) # EOF
        
    speaker_thread = threading.Thread(target=speaker_worker)
    start_time = time.time()
    speaker_thread.start()
    
    interrupted = False
    context_tokens = []
    current_kv_cache = None
    halt_timestamp = None
    
    prompt_template = """You are playing a game of progressive trivia. You receive clues about an entity token by token.
Current Clues: {context}

Question: Do you have enough high-confidence clues to definitively name the entity right now?
Answer EXACTLY 'STOP' or 'CONTINUE'."""

    while True:
        chunk = stream_queue.get()
        if chunk is None:
            break
        context_tokens.extend(chunk)
        context_text = tokenizer.decode(context_tokens)
        
        if mode == "dual":
            # 1. Base Stream Ingestion
            inputs = torch.tensor([chunk]).to("cuda")
            with torch.no_grad():
                base_outputs = model(inputs, past_key_values=current_kv_cache, use_cache=True)
                current_kv_cache = base_outputs.past_key_values
            
            # 2. Assessor Fork Pass
            assessor_inputs = tokenizer("\nClassification: ", return_tensors="pt").input_ids.to("cuda")
            t_eval = time.time()
            with torch.no_grad():
                assessor_out = model(assessor_inputs, past_key_values=current_kv_cache, use_cache=True)
                logits = assessor_out.logits[:, -1, :]
                s_logit = logits[0, stop_id].item()
                c_logit = logits[0, continue_id].item()
            
            if s_logit > c_logit:
                stop_event.set()
                halt_timestamp = time.time()
                interrupted = True
                break
        else:
            # Independent Decider:
            eval_prompt = prompt_template.format(context=context_text)
            inputs = tokenizer(eval_prompt, return_tensors="pt").input_ids.to("cuda")
            t_eval = time.time()
            with torch.no_grad():
                out = model(inputs, use_cache=True)
                logits = out.logits[:, -1, :]
                s_logit = logits[0, stop_id].item()
                c_logit = logits[0, continue_id].item()
            
            if s_logit > c_logit:
                stop_event.set()
                halt_timestamp = time.time()
                interrupted = True
                break
                
    speaker_thread.join()
    
    # Listener guesses
    if interrupted:
        guess_prompt = f"Description: {context_text}\nQuestion: Based on the description, what is the exact entity being described? (1-3 words only)\nAnswer: "
        guess_inputs = tokenizer(guess_prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            gen_out = model.generate(guess_inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
        guess = tokenizer.decode(gen_out[0][len(guess_inputs[0]):], skip_special_tokens=True).strip()
    else:
        guess = "NONE"
        
    total_consensus_time = time.time() - start_time
    time_to_halt = (halt_timestamp - start_time) if halt_timestamp else total_consensus_time
    
    return {
        "interrupted": interrupted,
        "tokens_read_by_listener": len(context_tokens),
        "tokens_emitted_by_speaker": len(speaker_tokens_emitted),
        "time_to_halt": time_to_halt,
        "total_consensus_time": total_consensus_time,
        "guess": guess
    }

# Run both on 10 scenarios
dual_times = []
indep_times = []
for idx, item in enumerate(scenarios):
    res_indep = run_scenario(item, mode="indep")
    res_dual = run_scenario(item, mode="dual")
    dual_times.append(res_dual["total_consensus_time"])
    indep_times.append(res_indep["total_consensus_time"])
    print(f"Scenario {idx:2d} | Indep Consensus: {res_indep['total_consensus_time']:.3f}s | Dual Consensus: {res_dual['total_consensus_time']:.3f}s | Diff: {res_dual['total_consensus_time'] - res_indep['total_consensus_time']:+.3f}s")

print(f"\nAverage Consensus Time across 10 scenarios:")
print(f"Independent Baseline: {sum(indep_times)/len(indep_times):.4f}s")
print(f"Dual-State Proposed:  {sum(dual_times)/len(dual_times):.4f}s")
