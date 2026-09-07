from transformers import AutoModelForCausalLM, AutoTokenizer
print("Loading Qwen via AutoModelForCausalLM...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
        trust_remote_code=True
    )
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
