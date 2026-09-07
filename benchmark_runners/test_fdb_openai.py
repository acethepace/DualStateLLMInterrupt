import sys, os, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
client = OpenAI()

data = json.load(open("datasets/Full-Duplex-Bench/v2/prompts_staged_200.json"))
task = data["splits"]["Daily"]["tasks"][0]

system_prompt = task["examiner_system_prompt"] + "\n" + task["examiner_task_prompt"]
messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Start the conversation. Only output your dialogue."}]

resp = client.chat.completions.create(model="gpt-4o", messages=messages)
t1 = resp.choices[0].message.content
print("[Examiner]:", t1)

base_stream = "I can certainly help you with that. Let me look up the available options."
print("[Examinee]:", base_stream)

messages.append({"role": "assistant", "content": t1})
messages.append({"role": "user", "content": f"The other speaker is saying: '{base_stream}'. Generate an interruption or backchannel."})
resp2 = client.chat.completions.create(model="gpt-4o", messages=messages)
print("[Examiner Interrupts]:", resp2.choices[0].message.content)
