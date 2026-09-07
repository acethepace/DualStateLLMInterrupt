import json
import time
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

with open("quizbowl_data/full.json") as f:
    dataset = json.load(f)[:5] # test first 5 scenarios

chunk_size = 5
chunk_delay = 0.075

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

model_name = "Qwen3-4B"
model_id = "Qwen/Qwen3-4B-Instruct-2507"
stop_tok = 45537
continue_tok = 16120

print(f"Loading {model_name}...")
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

            t0 = time.time()
            with torch.no_grad():
                out = model(inputs, use_cache=True)
                logits = out.logits[0, -1]
                s_val = logits[stop_tok].item()
                c_val = logits[continue_tok].item()
                halt_time = time.time() - t0

        total_halt_time += halt_time

        # Decision
        if s_val > c_val:
            # Buzzed!
            context_text = tokenizer.decode(context_tokens)
            guess_prompt = f"Clue: {context_text}\nQuestion: What is the exact entity being described? (1-3 words)\nAnswer:"
            guess_inp = tokenizer(guess_prompt, return_tensors="pt").input_ids.to("cuda")
            t_g0 = time.time()
            with torch.no_grad():
                gen_out = model.generate(guess_inp, max_new_tokens=6, pad_token_id=tokenizer.eos_token_id)
            g_time = time.time() - t_g0
            total_guess_time += g_time

            guess_str = tokenizer.decode(gen_out[0][len(guess_inp[0]):], skip_special_tokens=True).strip()
            actual = str(item['answer']).lower()
            is_corr = (actual in guess_str.lower()) or (guess_str.lower() in actual) if len(guess_str) > 1 else False

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
                # Wrong answer: game continues! Speaker resumes streaming!
                pass

    # End of stream final guess if not yet solved
    if not solved:
        context_text = tokenizer.decode(context_tokens)
        guess_prompt = f"Clue: {context_text}\nQuestion: What is the exact entity being described? (1-3 words)\nAnswer:"
        guess_inp = tokenizer(guess_prompt, return_tensors="pt").input_ids.to("cuda")
        t_g0 = time.time()
        with torch.no_grad():
            gen_out = model.generate(guess_inp, max_new_tokens=6, pad_token_id=tokenizer.eos_token_id)
        g_time = time.time() - t_g0
        total_guess_time += g_time
        guess_str = tokenizer.decode(gen_out[0][len(guess_inp[0]):], skip_special_tokens=True).strip()
        actual = str(item['answer']).lower()
        is_corr = (actual in guess_str.lower()) or (guess_str.lower() in actual) if len(guess_str) > 1 else False
        attempts.append({
            "attempt": len(attempts) + 1,
            "chunk_idx": "END_OF_STREAM",
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
        "attempts": attempts
    }

print("\n--- Running 5 Dual-State Scenarios ---")
dual_res = [run_scenario(item, mode="dual") for item in dataset]
for r in dual_res:
    print(f"ID: {r['scenario_id']} | Solved: {r['solved']} | Attempts: {r['num_attempts']} | Wrong: {r['num_wrong_buzzes']} | TTC: {r['consensus_time']:.2f}s")
    for a in r['attempts']:
        print(f"   [Attempt {a['attempt']} @ Tok {a['tokens_read']}] Guess: '{a['guess']}' (Corr: {a['is_correct']})")

print("\n--- Running 5 Independent Baseline Scenarios ---")
indep_res = [run_scenario(item, mode="indep") for item in dataset]
for r in indep_res:
    print(f"ID: {r['scenario_id']} | Solved: {r['solved']} | Attempts: {r['num_attempts']} | Reprefill Toks: {r['reprefill_tokens']} | TTC: {r['consensus_time']:.2f}s")
    for a in r['attempts']:
        print(f"   [Attempt {a['attempt']} @ Tok {a['tokens_read']}] Guess: '{a['guess']}' (Corr: {a['is_correct']})")

