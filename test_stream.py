import time
import torch
from dual_state_interruptor import setup_model, dual_state_stream_step

def simulate_streaming_interaction(
    model_name="unsloth/gemma-4-12b-bnb-4bit", 
    stop_word="STOP", 
    assessor_prompt="\n[INST] Based on the user's latest input, should we halt the current action? If yes, output only the exact word 'STOP'. Otherwise output 'CONTINUE'. [/INST]"
):
    print(f"Loading {model_name}...")
    model, tokenizer = setup_model(model_name)
    
    # Mocking a simulated audio/text stream from a user
    system_prompt = "System: You are an AI assistant. The user is streaming their voice."
    
    # Notice the 4th chunk contains a hard constraint reversal!
    user_speech_stream = [
        "Can you write a bash script to ",
        "delete all the files ",
        "in the /var/log directory? ",
        "Wait, no! Don't delete them! ",
        "Just list them out instead."
    ]
    
    print("\n--- Starting Dual-State Stream Simulation ---")
    
    # 1. Initialize the Base State with the system prompt
    input_ids = tokenizer(text=system_prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model(input_ids=input_ids, use_cache=True)
    base_kv_cache = outputs.past_key_values
    
    for i, chunk in enumerate(user_speech_stream):
        print(f"\n[Base Stream] Received chunk {i+1}: '{chunk}'")
        
        # Tokenize the new chunk of streaming speech
        chunk_ids = tokenizer(text=chunk, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        
        start_time = time.time()
        
        # 2. Run the Dual-State Step (Update Base -> Fork to Assessor)
        base_kv_cache, should_stop, assessor_response = dual_state_stream_step(
            model=model, 
            tokenizer=tokenizer, 
            new_tokens=chunk_ids, 
            base_kv_cache=base_kv_cache, 
            assessor_instruction=assessor_prompt
        )
        
        latency = time.time() - start_time
        print(f"[Assessor] Latency: {latency:.4f}s | Output: '{assessor_response.strip()}'")
        
        # 3. Handle Interruption
        if stop_word.upper() in assessor_response.upper():
            print(f"\n>>> 🚨 INTERRUPT TRIGGERED by Assessor on chunk {i+1}! Halting workflow.")
            # In a real app, this breaks the execution graph.
            break
            
    print("\n--- Stream Ended ---")

if __name__ == "__main__":
    # Parameters exposed here for easy modification during our upcoming ablation studies
    simulate_streaming_interaction(
        model_name="unsloth/gemma-4-12b-it",
        stop_word="STOP"
    )
