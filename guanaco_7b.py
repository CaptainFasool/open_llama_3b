import torch
import os
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define hyperparameters
max_seq_length = 512
max_output_length = 1024
num_beams = 16
length_penalty = 1.4
no_repeat_ngram_size = 2
temperature = 0.7
top_k = 150
top_p = 0.92
repetition_penalty = 2.1
early_stopping = True

# Load the pre-trained model and tokenizer
model_name = "huggyllama/llama-7b"
adapters_name = 'timdettmers/guanaco-7b'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
)

model = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.to("cuda")

# Define the chatbot function that saves the conversation in a file
filename = "conversation.txt"
if not os.path.exists(filename):
    open(filename, "w").close()  # create file if it doesn't exist

def chatbot(text):
    # Open the file in append mode
    with open(filename, "a+") as f:
        f.seek(0)  # go to the beginning of the file
        previous_conversation = f.read().splitlines()  # read previous conversation
        f.write(text + "\n")  # save user input

        # Format the prompt
        formatted_prompt = (
            f"A chat between a curious human and an artificial intelligence assistant."
            f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            f"### Human: {text} ### Assistant:"
        )

        # Tokenize the input text and convert to a PyTorch tensor
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        # Generate a response using the model
        outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=max_output_length, num_beams=num_beams, length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ngram_size, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, early_stopping=early_stopping)
        
        # Decode the response and return it
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        f.write(response + "\n")  # save chatbot response to file
        f.seek(0)  # go to the beginning of the file again
        updated_conversation = f.read().splitlines()  # read updated conversation

    return "\n".join(updated_conversation[len(previous_conversation):])

while True:
    text = input("You: ")
    print("ChatBot: " + chatbot(text))
