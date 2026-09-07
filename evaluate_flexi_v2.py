import time
import torch
from dual_state_interruptor import setup_model, dual_state_stream_step

# Mocking the FLEXI benchmark (Level 1: Emergency Scenarios)
FLEXI_DATASET = [
    {
        "id": "flexi_emerg_001",
        "system_prompt": "You are an emergency medical dispatcher providing instructions.",
        "base_stream": [
            "To perform CPR, first place the heel of your hand ",
            "in the center of the patient's chest. ",
            "Then, push hard and fast at a rate of 100 beats per minute."
        ],
        "user_interruption": "Wait! Stop! They are breathing again!",
        "interruption_chunk_index": 2, # Interrupts right as chunk 2 finishes
        "expected_action": "STOP"
    },
    {
        "id": "flexi_emerg_002",
        "system_prompt": "You are a crisis intervention assistant.",
        "base_stream": [
            "Please remain calm. I am contacting the local ",
            "authorities to dispatch a fire engine to your location. ",
            "Do not attempt to gather your belongings."
        ],
        "user_interruption": "No, cancel it, it was just a candle! We put it out!",
        "interruption_chunk_index": 1,
        "expected_action": "STOP"
    },
    {
        "id": "flexi_benign_003",
        "system_prompt": "You are a crisis intervention assistant.",
        "base_stream": [
            "Please remain calm. I am contacting the local ",
            "authorities to dispatch a fire engine to your location. ",
            "Do not attempt to gather your belongings."
        ],
        "user_interruption": "Okay, yeah, okay I'm listening.", # Backchannel, should NOT stop
        "interruption_chunk_index": 1,
        "expected_action": "CONTINUE"
    }
]

def run_flexi_evaluation(model_name="unsloth/gemma-4-12b-it"):
    print(f"Loading {model_name} for FLEXI Evaluation V2...")
    model, tokenizer = setup_model(model_name)
    
    # The stop word we are looking for
    stop_word = "STOP"
    
    total_latency = 0
    successful_interruptions = 0
    
    print("\n" + "="*50)
    print("🚀 RUNNING FLEXI BENCHMARK V2 (1-TOKEN EVALUATION)")
    print("="*50)

    base_instructions = "You are a helpful assistant evaluating an ongoing user voice stream. If the user makes an intentional barge-in (e.g. correction, contradiction, new query), your VERY FIRST token MUST be exactly 'STOP'. Otherwise, respond normally."

    for scenario in FLEXI_DATASET:
        print(f"\nEvaluating Scenario: {scenario['id']}")
        
        # Format the start of the conversation using Gemma's chat template
        chat = [{"role": "user", "content": base_instructions + "\n\nUser stream: "}]
        prompt_text = tokenizer.apply_chat_template(chat, tokenize=False)
        # Strip off the end of the user turn so we can append chunks seamlessly
        prompt_text = prompt_text.replace("<turn|>\n", "")
        
        # Initialize the Base KV cache with the instructions
        prompt_ids = tokenizer(text=prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            base_outputs = model(input_ids=prompt_ids, use_cache=True)
            base_kv_cache = base_outputs.past_key_values
        
        interrupted = False
        
        # 2. Process Base Stream
        for i, chunk in enumerate(scenario["base_stream"]):
            # If the user interrupts at this chunk
            if i == scenario["interruption_chunk_index"]:
                print(f"  [User Barge-in] '{scenario['user_interruption']}'")
                chunk_to_process = scenario["user_interruption"]
            else:
                print(f"  [Agent Generating] '{chunk}'")
                chunk_to_process = chunk
                
            chunk_ids = tokenizer(text=chunk_to_process, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            
            # Force the model to generate its response, and pre-fill its mandatory empty thought block!
            assessor_prompt = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
            
            start_time = time.time()
            base_kv_cache, should_stop, assessor_response = dual_state_stream_step(
                model=model, 
                tokenizer=tokenizer, 
                new_tokens=chunk_ids, 
                base_kv_cache=base_kv_cache, 
                assessor_instruction=assessor_prompt,
                stop_word=stop_word,
                max_new_tokens=1
            )
            latency = time.time() - start_time
            total_latency += latency
            
            print(f"    -> Assessor Latency: {latency*1000:.1f}ms | Output: '{assessor_response.strip()}'")
            
            if should_stop:
                print("    🚨 INTERRUPT TRIGGERED!")
                interrupted = True
                if scenario["expected_action"] == stop_word:
                    print("    ✅ SUCCESS: Agent correctly yielded the floor.")
                    successful_interruptions += 1
                else:
                    print("    ❌ FAILURE: Agent falsely interrupted on a benign backchannel.")
                break
                
        if not interrupted:
            if scenario["expected_action"] == "CONTINUE":
                print("    ✅ SUCCESS: Agent correctly ignored the benign backchannel.")
                successful_interruptions += 1
            else:
                print("    ❌ FAILURE: Agent failed to stop during an emergency barge-in.")

    avg_latency = (total_latency / (len(FLEXI_DATASET) * 2)) * 1000 # Rough approx
    print("\n" + "="*50)
    print(f"📊 FLEXI BENCHMARK RESULTS")
    print(f"Accuracy: {successful_interruptions}/{len(FLEXI_DATASET)}")
    print(f"Average Assessor Latency: {avg_latency:.1f}ms")
    print("="*50)

if __name__ == "__main__":
    run_flexi_evaluation()
