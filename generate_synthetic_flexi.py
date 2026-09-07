import json

base_templates = [
    {
        "id": "flexi_emerg_{}",
        "system_prompt": "You are an emergency medical dispatcher providing CPR instructions.",
        "base_stream": [
            "To perform CPR, first place the heel of your hand ",
            "in the center of the patient's chest. ",
            "Then, push hard and fast at a rate of 100 beats per minute."
        ],
        "user_interruption": "Wait! Stop! They are breathing again!",
        "interruption_chunk_index": 2,
        "expected_action": "STOP"
    },
    {
        "id": "flexi_benign_{}",
        "system_prompt": "You are a customer service AI describing a billing cycle.",
        "base_stream": [
            "Your billing cycle renews on the 15th of every month. ",
            "If you have an outstanding balance, it will be automatically ",
            "deducted from your saved payment method."
        ],
        "user_interruption": "Uh huh, okay.",
        "interruption_chunk_index": 1,
        "expected_action": "CONTINUE"
    },
    {
        "id": "flexi_correction_{}",
        "system_prompt": "You are a flight booking assistant confirming details.",
        "base_stream": [
            "I have your flight booked for Tuesday the 4th, departing ",
            "from JFK airport at 8:00 AM ",
            "and arriving in London at 8:00 PM."
        ],
        "user_interruption": "No, I said Wednesday the 5th!",
        "interruption_chunk_index": 0,
        "expected_action": "STOP"
    },
    {
        "id": "flexi_ambient_{}",
        "system_prompt": "You are a recipe assistant giving baking instructions.",
        "base_stream": [
            "Preheat the oven to 350 degrees. ",
            "While it heats up, mix the flour and sugar ",
            "in a large bowl until well combined."
        ],
        "user_interruption": "[Dog barking in the background]",
        "interruption_chunk_index": 1,
        "expected_action": "CONTINUE"
    },
    {
        "id": "flexi_clarification_{}",
        "system_prompt": "You are a technical support agent.",
        "base_stream": [
            "Please open the settings menu on your phone, ",
            "scroll down to the network and internet section, ",
            "and tap on the Wi-Fi option."
        ],
        "user_interruption": "Wait, where is the network section?",
        "interruption_chunk_index": 1,
        "expected_action": "STOP"
    }
]

dataset = []
for i in range(20):
    template = dict(base_templates[i % len(base_templates)])
    template["id"] = template["id"].format(i)
    dataset.append(template)

with open("datasets/flexi.json", "w") as f:
    json.dump(dataset, f, indent=4)
print("Generated datasets/flexi.json with 20 scenarios.")
