import json
import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

models = [
    ("Qwen3-4B", "Qwen/Qwen3-4B-Instruct-2507", 45537, 16120),
    ("Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B-Instruct", 46637, 16511),
    ("Gemma-4-12B", "unsloth/gemma-4-12b-it", 98174, 162216)
]

with open("quizbowl_data/full.json") as f:
    dataset = json.load(f) # All 200 scenarios

chunk_size = 5
chunk_delay = 0.075 # 75ms per 5-token chunk

system_prompt = """You are a Pictionary listener playing a competitive guessing game. The speaker streams clues describing a secret entity.
Your goal is to guess the entity correctly.

RULES:
1. Output 'CONTINUE' if the clue is incomplete, ambiguous, general, or if you do not know the exact answer with high certainty.
2. Output 'STOP' ONLY when you have enough specific clues to be confident in the exact entity name.
3. Premature wrong guesses incur a harsh penalty. When in doubt, ALWAYS output 'CONTINUE'.

EXAMPLES:
Clue: In the 1920s, a tiger, a kangaroo,
Classification: CONTINUE

Clue: In the 1920s, a tiger, a kangaroo, a donkey, a pig, and a bear were bought from Harrods in London. Who was this buyer?
Classification: STOP

Clue: Which political party governs
Classification: CONTINUE

Clue: Which political party governs the country directly south of Botswana?
Classification: STOP

Clue: How many non-pet characters live in
Classification: CONTINUE

Clue: How many non-pet characters live in SpongeBob's neighborhood?
Classification: STOP

Clue: """

all_results = {}

for model_name, model_id, stop_tok, continue_tok in models:
    print(f"\n========================================================")
    print(f"Running Multi-Attempt HANDRAISER for {model_name}...")
    print(f"========================================================")

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
        attempts = []
        solved = False
        speaker_elapsed = 0.0
        total_halt_time = 0.0
        total_guess_time = 0.0
        reprefill_tokens = 0
        prompt_tokens_eval = 0

        for i in range(0, len(question_tokens), chunk_size):
            chunk = question_tokens[i:i+chunk_size]
            context_tokens.extend(chunk)
            speaker_elapsed += chunk_delay

            halt_time = 0.0
            if mode == "dual":
                with torch.no_grad():
                    base_out = model(torch.tensor([chunk]).to("cuda"), past_key_values=current_kv, use_cache=True)
                    current_kv = base_out.past_key_values

                    assessor_inp = tokenizer("\nClassification:", return_tensors="pt").input_ids.to("cuda")
                    prompt_tokens_eval += len(assessor_inp[0])
                    t0 = time.time()
                    assessor_out = model(assessor_inp, past_key_values=current_kv, use_cache=True)
                    logits = assessor_out.logits[0, -1]
                    s_val = logits[stop_tok].item()
                    c_val = logits[continue_tok].item()
                    halt_time = time.time() - t0
            else:
                context_text = tokenizer.decode(context_tokens)
                full_prompt = system_prompt + context_text + "\nClassification:"
                inputs = tokenizer(full_prompt, return_tensors="pt").input_ids.to("cuda")
                reprefill_tokens += len(inputs[0])
                prompt_tokens_eval += len(sys_tokens) + 2

                t0 = time.time()
                with torch.no_grad():
                    out = model(inputs, use_cache=True)
                    logits = out.logits[0, -1]
                    s_val = logits[stop_tok].item()
                    c_val = logits[continue_tok].item()
                    halt_time = time.time() - t0

            total_halt_time += halt_time

            # Interruption check
            if s_val > c_val:
                context_text = tokenizer.decode(context_tokens)
                messages = [
                    {"role": "user", "content": f"{context_text}\nAnswer immediately with ONLY the name of the entity, no explanation."}
                ]
                try:
                    guess_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    guess_prompt = f"Clue: {context_text}\nAnswer with ONLY the entity name directly:\n"
                guess_inp = tokenizer(guess_prompt, return_tensors="pt").input_ids.to("cuda")
                
                t_g0 = time.time()
                with torch.no_grad():
                    gen_out = model.generate(guess_inp, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
                g_time = time.time() - t_g0
                total_guess_time += g_time

                guess_str = tokenizer.decode(gen_out[0][len(guess_inp[0]):], skip_special_tokens=True).strip()
                actual = str(item['answer']).lower().strip()
                g_low = guess_str.lower().strip()
                clean_actual = actual.replace('.', '').replace('-', ' ')
                clean_guess = g_low.replace('.', '').replace('-', ' ')
                is_corr = (clean_actual in clean_guess) or (clean_guess in clean_actual) if len(clean_guess) > 1 else False

                attempt_obj = {
                    "attempt": len(attempts) + 1,
                    "chunk_idx": i // chunk_size + 1,
                    "tokens_read": len(context_tokens),
                    "clue_so_far": context_text,
                    "guess": guess_str,
                    "target_answer": item['answer'],
                    "is_correct": is_corr,
                    "halt_time": halt_time
                }
                attempts.append(attempt_obj)

                if is_corr:
                    solved = True
                    break
                else:
                    # Wrong buzz: speaker resumes streaming!
                    pass

        # End of stream final guess if not yet solved
        if not solved:
            context_text = tokenizer.decode(context_tokens)
            messages = [
                {"role": "user", "content": f"{context_text}\nAnswer immediately with ONLY the name of the entity, no explanation."}
            ]
            try:
                guess_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                guess_prompt = f"Clue: {context_text}\nAnswer with ONLY the entity name directly:\n"
            guess_inp = tokenizer(guess_prompt, return_tensors="pt").input_ids.to("cuda")
            
            t_g0 = time.time()
            with torch.no_grad():
                gen_out = model.generate(guess_inp, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
            g_time = time.time() - t_g0
            total_guess_time += g_time
            guess_str = tokenizer.decode(gen_out[0][len(guess_inp[0]):], skip_special_tokens=True).strip()
            actual = str(item['answer']).lower().strip()
            g_low = guess_str.lower().strip()
            clean_actual = actual.replace('.', '').replace('-', ' ')
            clean_guess = g_low.replace('.', '').replace('-', ' ')
            is_corr = (clean_actual in clean_guess) or (clean_guess in clean_actual) if len(clean_guess) > 1 else False

            attempts.append({
                "attempt": len(attempts) + 1,
                "chunk_idx": "EOS",
                "tokens_read": len(context_tokens),
                "clue_so_far": context_text,
                "guess": guess_str,
                "target_answer": item['answer'],
                "is_correct": is_corr,
                "halt_time": 0.0
            })
            if is_corr:
                solved = True

        consensus_time = speaker_elapsed + total_halt_time + total_guess_time
        num_wrong = sum(1 for a in attempts if not a['is_correct'])

        return {
            "scenario_id": item['id'],
            "question": item['question'],
            "target_answer": item['answer'],
            "solved": solved,
            "tokens_read_to_solve": attempts[-1]['tokens_read'] if solved else len(question_tokens),
            "total_stream_tokens": len(question_tokens),
            "num_attempts": len(attempts),
            "num_wrong_buzzes": num_wrong,
            "consensus_time": consensus_time,
            "reprefill_tokens": reprefill_tokens,
            "prompt_tokens_eval": prompt_tokens_eval,
            "attempts": attempts
        }

    model_results = {}
    for m in ["indep", "dual"]:
        res_list = []
        print(f"--- Running {m.upper()} mode for {model_name} (200 scenarios) ---")
        for idx, item in enumerate(dataset):
            res = run_scenario(item, mode=m)
            res_list.append(res)
            if (idx + 1) % 50 == 0:
                solved_so_far = sum(1 for x in res_list if x['solved'])
                print(f"  [{m.upper()}] Processed {idx+1}/200 scenarios | Solved: {solved_so_far}/{idx+1} ({solved_so_far/(idx+1)*100:.1f}%)")

        # Aggregate metrics
        solved_count = sum(1 for x in res_list if x['solved'])
        resolution_acc = solved_count / len(res_list) * 100
        
        # First attempt precision
        first_attempts = [x['attempts'][0] for x in res_list if len(x['attempts']) > 0]
        first_buzz_prec = sum(1 for a in first_attempts if a['is_correct']) / len(first_attempts) * 100 if first_attempts else 0.0

        interrupted_attempts = [a for x in res_list for a in x['attempts'] if a['chunk_idx'] != "EOS"]
        avg_tth = sum(a['halt_time'] for a in interrupted_attempts) / len(interrupted_attempts) if interrupted_attempts else 0.0
        avg_ttc = sum(x['consensus_time'] for x in res_list) / len(res_list)
        avg_toks_to_solve = sum(x['tokens_read_to_solve'] for x in res_list) / len(res_list)
        avg_reprefill = sum(x['reprefill_tokens'] for x in res_list) / len(res_list)
        avg_prompt_tokens = sum(x['prompt_tokens_eval'] for x in res_list) / len(res_list)
        avg_stream_toks = sum(x['total_stream_tokens'] for x in res_list) / len(res_list)
        avg_wrong_buzzes = sum(x['num_wrong_buzzes'] for x in res_list) / len(res_list)

        # Multi-attempt game scores for beta in [2, 3, 5]
        # S = 10 + 10 * (stream_len - solve_tok) / stream_len if solved else 0 - beta * wrong_buzzes
        scores_by_beta = {}
        for beta in [2, 3, 5]:
            scenario_scores = []
            for x in res_list:
                s = 0.0
                if x['solved']:
                    speed_frac = max(0.0, (x['total_stream_tokens'] - x['tokens_read_to_solve']) / x['total_stream_tokens'])
                    s += 10.0 + 10.0 * speed_frac
                s -= beta * x['num_wrong_buzzes']
                scenario_scores.append(s)
            scores_by_beta[f"beta_{beta}"] = round(sum(scenario_scores) / len(scenario_scores), 2)

        model_results[m] = {
            "resolution_accuracy": round(resolution_acc, 2),
            "first_buzz_precision": round(first_buzz_prec, 2),
            "time_to_halt_s": round(avg_tth, 4),
            "consensus_time_s": round(avg_ttc, 4),
            "avg_tokens_to_solve": round(avg_toks_to_solve, 1),
            "avg_stream_tokens": round(avg_stream_toks, 1),
            "avg_wrong_buzzes": round(avg_wrong_buzzes, 2),
            "context_reprefill_tokens": round(avg_reprefill, 1) if m == "indep" else 0.0,
            "prompt_tokens_eval": round(avg_prompt_tokens, 1),
            "game_scores": scores_by_beta,
            "scenarios": res_list
        }

    all_results[model_name] = model_results
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

with open("full_duplex_handraiser_official_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n========================================================")
print("Multi-Attempt Full-Duplex HANDRAISER Complete!")
print("Results saved to full_duplex_handraiser_official_results.json")
print("========================================================")
