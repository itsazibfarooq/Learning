import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama_Storyteller", torch_dtype=torch.bfloat16, device_map="auto")
messages = [
    {"role": "user", "content": "A young boy running in the jungle"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
outputs = pipe(prompt, max_new_tokens=512)
print(outputs[0]["generated_text"])
