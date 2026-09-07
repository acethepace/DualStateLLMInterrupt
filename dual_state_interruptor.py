import torch
from unsloth import FastLanguageModel

def setup_model(model_name="unsloth/gemma-4-12b-bnb-4bit"):
    """
    Initialize the model via Unsloth for fast, memory-efficient local inference.
    """
    max_seq_length = 4096
    if "llama" in model_name.lower() or "qwen" in model_name.lower():
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_4bit=True)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {model_name} via standard transformers...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto", quantization_config=quant_config
        )
        return model, tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def dual_state_stream_step(model, tokenizer, new_tokens, base_kv_cache, assessor_instruction, stop_word="<|Start-Listening|>", max_new_tokens=15, version="v2"):
    with torch.no_grad():
        base_outputs = model(
            input_ids=new_tokens,
            past_key_values=base_kv_cache,
            use_cache=True
        )
    updated_base_kv = base_outputs.past_key_values
    
    assessor_input_ids = tokenizer(text=assessor_instruction, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    
    assessor_text = ""
    current_input_ids = assessor_input_ids
    current_kv = updated_base_kv
    
    assessor_confidence = 0.0

    if version in ["v3", "v4"]:
        # Logit Bias / Forced Classification
        with torch.no_grad():
            assessor_outputs = model(
                input_ids=current_input_ids,
                past_key_values=current_kv, 
                use_cache=True
            )
        
        logits = assessor_outputs.logits[:, -1, :]
        
        # STOP tokens: 'STOP': 51769, ' STOP': 46637
        # CONTINUE tokens: 'CONTIN': 24194, ' CONTIN': 16511
        stop_ids = [50669] if 'Qwen' in model.name_or_path else [51769, 46637]
        cont_ids = [35045] if 'Qwen' in model.name_or_path else [24194, 16511]
        
        stop_logits = logits[0, stop_ids]
        cont_logits = logits[0, cont_ids]
        
        # Create a tiny softmax over just these 4 tokens
        combined_logits = torch.cat([stop_logits, cont_logits])
        probs = torch.nn.functional.softmax(combined_logits, dim=-1)
        
        stop_prob = probs[0].item() if 'Qwen' in model.name_or_path else probs[0].item() + probs[1].item()
        cont_prob = probs[1].item() if 'Qwen' in model.name_or_path else probs[2].item() + probs[3].item()
        
        assessor_confidence = stop_prob
        should_stop = stop_prob > cont_prob
        
        assessor_text = "STOP" if should_stop else "CONTINUE"
        
        return updated_base_kv, should_stop, assessor_text, assessor_confidence
    else:
        # Standard autoregressive loop
        for _ in range(max_new_tokens):
            with torch.no_grad():
                assessor_outputs = model(
                    input_ids=current_input_ids,
                    past_key_values=current_kv, 
                    use_cache=True
                )
            
            current_kv = assessor_outputs.past_key_values
            next_token_id = torch.argmax(assessor_outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            token_str = tokenizer.decode(next_token_id[0])
            assessor_text += token_str
            current_input_ids = next_token_id
            
            if next_token_id[0].item() in [tokenizer.eos_token_id, 128009, 128001]:
                break
                
            if stop_word.upper() in assessor_text.upper() or "CONTINUE" in assessor_text.upper() or "PROCEED" in assessor_text.upper():
                break
        
        should_stop = stop_word.upper() in assessor_text.upper()
        
        return updated_base_kv, should_stop, assessor_text, assessor_confidence

if __name__ == "__main__":
    print("Dual-State LLM Scaffold Initialized.")
    # We will build out the simulated streaming loop here next.
