from transformers import AutoModelForCausalLM, AutoTokenizer
print("Loading via AutoModelForCausalLM...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gemma-4-12b-it",
        trust_remote_code=True
    )
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
