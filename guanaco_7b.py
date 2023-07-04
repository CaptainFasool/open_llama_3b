from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
model_name = 'KBlueLeaf/guanaco-7b-leh-v2'
filename = "conversation.txt"

def generate_text(model, tokenizer, prompt):
    # Encode the input and send to device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate response
    output = model.generate(input_ids, 
                            max_length=1024, 
                            temperature=0.65, 
                            top_k=50, 
                            top_p=0.95)
    
    # Decode the response
    output_text = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return output_text

def chatbot(model, tokenizer, text):
    # Open the file in append mode
    with open(filename, "a+") as f:
        f.seek(0)  # go to the beginning of the file
        previous_conversation = f.read().splitlines()  # read previous conversation
        f.write(text + "\n")  # save user input

        # Generate a response using the model
        response = generate_text(model, tokenizer, text)
        f.write(response + "\n")  # save chatbot response to file
        f.seek(0)  # go to the beginning of the file again
        updated_conversation = f.read().splitlines()  # read updated conversation

    return "\n".join(updated_conversation[len(previous_conversation):])

def main():
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    while True:
        prompt = input("Please enter a prompt: ")
        response = chatbot(model, tokenizer, prompt)
        print("Generated Text:")
        print(response)

if __name__ == "__main__":
    main()
