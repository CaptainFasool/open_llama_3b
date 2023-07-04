import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

def generate_text(model_name, prompt):
    # Set up the model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Make sure the model uses GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text
    output = model.generate(input_ids, max_length=1000, do_sample=True,
                            top_p=0.92, top_k=30, temperature=0.6,
                            pad_token_id=tokenizer.eos_token_id)

    # Decode the output
    return tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)


filename = "conversation.txt"
if not os.path.exists(filename):
    open(filename, "w").close()  # create file if it doesn't exist

def main():
    model_name = 'KBlueLeaf/guanaco-7b-leh-v2'

    while True:
        # Open the file in append mode
        with open(filename, "a+") as f:
            f.seek(0)  # go to the beginning of the file
            previous_conversation = f.read().splitlines()  # read previous conversation

            prompt = input("Please enter a prompt: ")
            f.write('User: ' + prompt + "\n")  # save user input
            print('Generated Text:')
            response = generate_text(model_name, prompt)
            f.write('Assistant: ' + response + "\n")  # save chatbot response to file
            print(response)
            print('\n')

            f.seek(0)  # go to the beginning of the file again
            updated_conversation = f.read().splitlines()  # read updated conversation

        print("\n".join(updated_conversation[len(previous_conversation):]))


if __name__ == '__main__':
    main()
