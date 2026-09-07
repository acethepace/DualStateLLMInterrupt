import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model...")
model_name = "unsloth/gemma-4-12b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

def test_prompt(instruction, text_stream):
    chat = [{"role": "user", "content": instruction + "\n\nUser stream: "}]
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False)
    prompt_text = prompt_text.replace("<turn|>\n", "")
    
    full_text = prompt_text + text_stream + "<turn|>\n<|turn>model\n"
    
    input_ids = tokenizer(text=full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    
    print("\n--- Testing ---")
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=10, do_sample=False)
    
    result = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
    print(f"Output: {result!r}")

# Test 1: with <|Start-Listening|>
test_prompt(
    "You are evaluating a voice stream. If the user makes an intentional barge-in (e.g. correction, contradiction), output exactly `<|Start-Listening|>`. Otherwise act normally.",
    "To perform CPR... Wait! Stop! They are breathing again!"
)

# Test 2: with [START_LISTENING]
test_prompt(
    "You are evaluating a voice stream. If the user makes an intentional barge-in (e.g. correction, contradiction), output exactly `[START_LISTENING]`. Otherwise act normally.",
    "To perform CPR... Wait! Stop! They are breathing again!"
)
