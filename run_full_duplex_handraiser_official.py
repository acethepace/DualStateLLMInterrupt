import json
import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

models = [
    ("Qwen3-4B", "Qwen/Qwen3-4B-Instruct-2507", 45537, 54390),
    ("Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B-Instruct", 46637, 55490),
    ("Gemma-4-12B", "unsloth/gemma-4-12b-it", 98174, 213110)
]

with open("quizbowl_data/full.json") as f:
    dataset = json.load(f) # All 200 scenarios

chunk_size = 5
chunk_delay = 0.075 # 75ms per 5-token chunk (15ms / token)

system_prompt = """You are a Pictionary listener. The speaker is describing a target entity.
You must NOT interrupt unless you have enough specific clues to uniquely identify the entity.

RULES:
- If the description is vague (e.g. "This famous man", "It is a structure located in", "How many"), output EXACTLY 'WAIT'.
- If the description gives away a highly specific clue that guarantees the answer, output EXACTLY 'STOP'.

EXAMPLES:
Description so far: This famous
Classification: WAIT

Description so far: This famous landmark is an iron lattice tower located in Paris
Classification: STOP

Description so far: Which political party governs
Classification: WAIT

Description so far: """

results_all = {}

for model_name, model_id, stop_id, wait_id in models:
    print(f"\n==========================================")
    print(f"Running Full-Duplex HANDRAISER for {model_name}...")
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
    
    sys_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
    
    def run_scenario(item, mode="dual"):
        question_tokens = tokenizer.encode(item['question'], add_special_tokens=False)
        
        current_kv = None
        if mode == "dual":
            with torch.no_grad():
                base_out = model(torch.tensor([sys_tokens]).to("cuda"), use_cache=True)
                current_kv = base_out.past_key_values
                
        context_tokens = []
        interrupted = False
        halt_time = 0.0
        speaker_elapsed = 0.0
        reprefill_tokens = 0
        
        for i in range(0, len(question_tokens), chunk_size):
            chunk = question_tokens[i:i+chunk_size]
            context_tokens.extend(chunk)
            speaker_elapsed += chunk_delay
            
            if mode == "dual":
                with torch.no_grad():
                    base_out = model(torch.tensor([chunk]).to("cuda"), past_key_values=current_kv, use_cache=True)
                    current_kv = base_out.past_key_values
                    
                    assessor_inp = tokenizer("\nClassification:", return_tensors="pt").input_ids.to("cuda")
                    t0 = time.time()
                    assessor_out = model(assessor_inp, past_key_values=current_kv, use_cache=True)
                    logits = assessor_out.logits[0, -1]
                    s_val = logits[stop_id].item()
                    w_val = logits[wait_id].item()
                    halt_time = time.time() - t0
                
                reprefill_tokens += 0
                if s_val > w_val:
                    interrupted = True
                    break
            else:
                context_text = tokenizer.decode(context_tokens)
                full_prompt = system_prompt + context_text + "\nClassification:"
                inputs = tokenizer(full_prompt, return_tensors="pt").input_ids.to("cuda")
                reprefill_tokens += len(inputs[0])
                
                t0 = time.time()
                with torch.no_grad():
                    out = model(inputs, use_cache=True)
                    logits = out.logits[0, -1]
                    s_val = logits[stop_id].item()
                    w_val = logits[wait_id].item()
                    halt_time = time.time() - t0
                    
                if s_val > w_val:
                    interrupted = True
                    break
                    
        # Guess generation
        context_text = tokenizer.decode(context_tokens)
        if interrupted:
            guess_prompt = f"Description: {context_text}\nQuestion: What is the exact entity being described? (1-3 words)\nAnswer:"
            guess_inp = tokenizer(guess_prompt, return_tensors="pt").input_ids.to("cuda")
            t_guess_0 = time.time()
            with torch.no_grad():
                gen_out = model.generate(guess_inp, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
            guess_time = time.time() - t_guess_0
            guess = tokenizer.decode(gen_out[0][len(guess_inp[0]):], skip_special_tokens=True).strip()
        else:
            guess_time = 0.0
            guess = "NONE"
            
        consensus_time = speaker_elapsed + halt_time + guess_time
        actual = str(item['answer']).lower()
        is_correct = (actual in guess.lower()) or (guess.lower() in actual) if len(guess) > 1 else False
        
        return {
            "interrupted": interrupted,
            "tokens_read": len(context_tokens),
            "halt_time": halt_time,
            "consensus_time": consensus_time,
            "is_correct": is_correct,
            "reprefill_tokens": reprefill_tokens
        }

    model_summary = {}
    for m in ["indep", "dual"]:
        res_list = []
        for idx, item in enumerate(dataset):
            res_list.append(run_scenario(item, mode=m))
            if (idx + 1) % 50 == 0:
                print(f"[{model_name} {m.upper()}] Processed {idx+1}/200 scenarios...")
                
        interrupted_subset = [x for x in res_list if x['interrupted']]
        avg_tth = sum(x['halt_time'] for x in interrupted_subset) / len(interrupted_subset) if interrupted_subset else 0.0
        avg_ttc = sum(x['consensus_time'] for x in res_list) / len(res_list)
        avg_toks = sum(x['tokens_read'] for x in res_list) / len(res_list)
        avg_reprefill = sum(x['reprefill_tokens'] for x in res_list) / len(res_list)
        accuracy = sum(1 for x in res_list if x['is_correct']) / len(res_list) * 100
        passive_rate = sum(1 for x in res_list if not x['interrupted']) / len(res_list) * 100
        
        model_summary[m] = {
            "interruption_accuracy": round(accuracy, 2),
            "passive_listening_rate": round(passive_rate, 2),
            "time_to_halt_s": round(avg_tth, 4),
            "context_reprefill_tokens": round(avg_reprefill, 1) if m == "indep" else 0.0,
            "consensus_time_s": round(avg_ttc, 4),
            "avg_tokens_read": round(avg_toks, 1)
        }
        
    speedup = (model_summary["indep"]["consensus_time_s"] - model_summary["dual"]["consensus_time_s"]) / model_summary["indep"]["consensus_time_s"] * 100
    tth_speedup = (model_summary["indep"]["time_to_halt_s"] - model_summary["dual"]["time_to_halt_s"]) / model_summary["indep"]["time_to_halt_s"] * 100
    print(f"\nResults for {model_name}:")
    print(f"  Indep: TTH={model_summary['indep']['time_to_halt_s']}s, TTC={model_summary['indep']['consensus_time_s']}s, Acc={model_summary['indep']['interruption_accuracy']}%, Tokens={model_summary['indep']['avg_tokens_read']}")
    print(f"  Dual:  TTH={model_summary['dual']['time_to_halt_s']}s, TTC={model_summary['dual']['consensus_time_s']}s, Acc={model_summary['dual']['interruption_accuracy']}%, Tokens={model_summary['dual']['avg_tokens_read']}")
    print(f"  Consensus Time Speedup: {speedup:+.1f}% | Time-to-Halt Reduction: {tth_speedup:+.1f}%")
    
    results_all[model_name] = model_summary
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

with open("full_duplex_handraiser_official_results.json", "w") as f:
    json.dump(results_all, f, indent=2)

print("\nOfficial Full-Duplex HANDRAISER Benchmark Completed Successfully!")
