import json
import os
from datasets import load_dataset

def clean_and_save():
    os.makedirs('quizbowl_data', exist_ok=True)
    
    final_dataset = []
    
    # 1. AdvQA
    print("Loading AdvQA...")
    try:
        advqa = load_dataset("qanta-challenge/AdvQA", "advqa")
        split = 'eval' if 'eval' in advqa else list(advqa.keys())[0]
        for i, item in enumerate(advqa[split]):
            if i >= 100:
                break
            final_dataset.append({
                "id": f"advqa_{i}",
                "question": item['question'],
                "answer": item['answer']
            })
    except Exception as e:
        print(f"Failed to load AdvQA: {e}")

    # 2. Protobowl
    print("Loading Protobowl...")
    try:
        protobowl = load_dataset("mgor/protobowl-11-13", "progressive-clues")
        split = 'eval' if 'eval' in protobowl else list(protobowl.keys())[0]
        for i, item in enumerate(protobowl[split]):
            if i >= 100:
                break
            # Handle list of answers
            answer = item['clean_answers'][0] if isinstance(item['clean_answers'], list) and len(item['clean_answers']) > 0 else item['clean_answers']
            final_dataset.append({
                "id": f"protobowl_{i}",
                "question": item['full_quiz_question'],
                "answer": answer
            })
    except Exception as e:
        print(f"Failed to load Protobowl: {e}")

    # Save small subset
    with open('quizbowl_data/subset.json', 'w') as f:
        json.dump(final_dataset[:20], f, indent=2)
        
    # Save larger set
    with open('quizbowl_data/full.json', 'w') as f:
        json.dump(final_dataset, f, indent=2)
        
    print(f"Saved {len(final_dataset[:20])} to subset.json and {len(final_dataset)} to full.json")

if __name__ == "__main__":
    clean_and_save()
