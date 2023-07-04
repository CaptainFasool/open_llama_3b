import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(prompt):
    # Specify the model path
    model_path = 'openlm-research/open_llama_7b'

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Convert the prompt to model inputs
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    output = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1)

    # Decode the output and return the text
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Infinite loop to continually ask for prompts
while True:
    # Get user input
    prompt = input("Please enter a prompt: ")
    
    # If the user types 'quit', break the loop
    if prompt.lower() == 'quit':
        break
    
    # Generate text based on the prompt and print it
    print(generate_text(prompt))
