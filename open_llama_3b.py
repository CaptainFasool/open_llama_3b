import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# Choose the model path according to which model you want to load
model_path = 'openlm-research/open_llama_3b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model.to(device)  # move the model to GPU if available

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to(device)  # move the input to GPU if available

# Generate text
generation_output = model.generate(input_ids, max_new_tokens=32)
print(tokenizer.decode(generation_output[0]))
