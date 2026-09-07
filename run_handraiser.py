import json
import time
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def run_experiment(model_id, dataset_path, is_dual_state=False):
    print(f"Loading {model_id}...")
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    prompt_template = """You are a Pictionary listener. The speaker is describing a target entity.
You must NOT interrupt unless you have enough specific clues to uniquely identify the entity.

RULES:
- If the description is vague (e.g. "This famous man", "It is a structure located in", "How many"), output EXACTLY 'CONTINUE'.
- If the description gives away a highly specific clue that guarantees the answer, output EXACTLY 'STOP'.

EXAMPLES:
Description so far: This famous
Classification: CONTINUE

Description so far: This famous landmark is an iron lattice tower located in Paris
Classification: STOP

Description so far: Which political party governs
Classification: CONTINUE

Description so far: {context}
Classification: """

    stop_id = tokenizer.encode("STOP", add_special_tokens=False)
    stop_id = stop_id[-1] if isinstance(stop_id, list) and len(stop_id) > 0 else stop_id
    
    continue_id = tokenizer.encode("CONTINUE", add_special_tokens=False)
    continue_id = continue_id[-1] if isinstance(continue_id, list) and len(continue_id) > 0 else continue_id

    results = []
    
    for item in dataset:
        question_tokens = tokenizer.encode(item['question'], add_special_tokens=False)
        context_tokens = []
        interrupted = False
        guess = ""
        total_eval_latency = 0
        redundant_tokens = 0
        halt_latency = 0
        
        current_kv_cache = None
        chunk_size = 5
        start_time = time.time()
        
        for i in range(0, len(question_tokens), chunk_size):
            chunk = question_tokens[i:i+chunk_size]
            context_tokens.extend(chunk)
            
            context_text = tokenizer.decode(context_tokens)
            eval_prompt = prompt_template.format(context=context_text)
            
            if is_dual_state:
                with torch.no_grad():
                    inputs = torch.tensor([chunk]).to(model.device)
                    base_outputs = model(inputs, past_key_values=current_kv_cache, use_cache=True)
                    current_kv_cache = base_outputs.past_key_values
                
                assessor_prompt = "\nClassification: "
                assessor_inputs = tokenizer(assessor_prompt, return_tensors="pt").input_ids.to(model.device)
                
                eval_start = time.time()
                with torch.no_grad():
                    assessor_outputs = model(assessor_inputs, past_key_values=current_kv_cache, use_cache=True)
                    logits = assessor_outputs.logits[:, -1, :]
                    
                    stop_logit = logits[0, stop_id].item()
                    continue_logit = logits[0, continue_id].item()
                    
                eval_time = time.time() - eval_start
                total_eval_latency += eval_time
                redundant_tokens += 0 
            else:
                inputs = tokenizer(eval_prompt, return_tensors="pt").input_ids.to(model.device)
                
                eval_start = time.time()
                with torch.no_grad():
                    outputs = model(inputs, use_cache=True)
                    logits = outputs.logits[:, -1, :]
                    
                    stop_logit = logits[0, stop_id].item()
                    continue_logit = logits[0, continue_id].item()
                    
                eval_time = time.time() - eval_start
                total_eval_latency += eval_time
                redundant_tokens += len(inputs[0]) 
                
            if stop_logit > continue_logit:
                halt_latency = eval_time
                interrupted = True
                break
                
        if interrupted:
            guess_prompt = f"Description: {context_text}\nQuestion: Based on the description, what is the exact entity being described? (1-3 words only)\nAnswer: "
            guess_inputs = tokenizer(guess_prompt, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                gen_out = model.generate(guess_inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
            guess = tokenizer.decode(gen_out[0][len(guess_inputs[0]):], skip_special_tokens=True).strip()
            
        total_time = time.time() - start_time
        
        results.append({
            "id": item['id'],
            "interrupted": interrupted,
            "tokens_consumed": len(context_tokens),
            "total_tokens": len(question_tokens),
            "guess": guess,
            "actual_answer": item['answer'],
            "context_at_interrupt": context_text,
            "redundant_tokens_processed": redundant_tokens,
            "time_to_halt": halt_latency,
            "total_time_to_consensus": total_time
        })
        
    mode_str = "dual" if is_dual_state else "indep"
    safe_name = model_id.split("/")[-1]
    with open(f"quizbowl_results_{safe_name}_{mode_str}.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    model_id = sys.argv[1]
    # For speed, test on the subset first
    run_experiment(model_id, "quizbowl_data/full.json", is_dual_state=False)
    run_experiment(model_id, "quizbowl_data/full.json", is_dual_state=True)
