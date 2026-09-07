with open("benchmark_runners/fdb_runner.py", "r") as f:
    content = f.read()

old_code = """            messages.append({"role": "assistant", "content": examiner_turn_1})
            messages.append({"role": "user", "content": f"The other speaker is currently saying: '{''.join(base_stream)}'. Generate an interruption or a backchannel (like 'uh huh') that you would say while they are speaking. Output ONLY the interruption text."})
            
            resp2 = client.chat.completions.create(model="gpt-5.6-luna", messages=messages)
            interruption = resp2.choices[0].message.content
            
            expected_action = "STOP" if len(interruption.split()) > 3 else "CONTINUE"
"""

new_code = """            messages.append({"role": "assistant", "content": examiner_turn_1})
            messages.append({"role": "user", "content": f"The other speaker is currently saying: '{''.join(base_stream)}'. Generate an interruption or a backchannel (like 'uh huh') that you would say while they are speaking. Output a JSON object with two keys: 'type' (either 'STOP' for a real interruption, or 'CONTINUE' for a benign backchannel/filler) and 'text' (the actual interruption text). Output ONLY valid JSON."})
            
            resp2 = client.chat.completions.create(model="gpt-5.6-luna", messages=messages, response_format={"type": "json_object"})
            try:
                parsed = json.loads(resp2.choices[0].message.content)
                interruption = parsed["text"]
                expected_action = parsed["type"]
            except Exception:
                interruption = resp2.choices[0].message.content
                expected_action = "CONTINUE"
"""

content = content.replace(old_code, new_code)
with open("benchmark_runners/fdb_runner.py", "w") as f:
    f.write(content)
