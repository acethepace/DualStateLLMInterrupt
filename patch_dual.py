import re

with open("dual_state_interruptor.py", "r") as f:
    content = f.read()

def_start = content.find("def dual_state_stream_step")
def_end = content.find("if __name__ == \"__main__\":")

new_func = """def dual_state_stream_step(model, tokenizer, new_tokens, base_kv_cache, assessor_instruction, stop_word="<|Start-Listening|>", max_new_tokens=15, version="v2"):
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

    if version == "v3":
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
        stop_ids = [51769, 46637]
        cont_ids = [24194, 16511]
        
        stop_logits = logits[0, stop_ids]
        cont_logits = logits[0, cont_ids]
        
        # Create a tiny softmax over just these 4 tokens
        combined_logits = torch.cat([stop_logits, cont_logits])
        probs = torch.nn.functional.softmax(combined_logits, dim=-1)
        
        stop_prob = probs[0].item() + probs[1].item()
        cont_prob = probs[2].item() + probs[3].item()
        
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

"""

content = content[:def_start] + new_func + content[def_end:]

with open("dual_state_interruptor.py", "w") as f:
    f.write(content)
print("done")
