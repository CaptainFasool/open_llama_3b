import os
import torch
from transformers import pipeline, AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = 'mosaicml/mpt-7b-chat'
tokenizer_name = "EleutherAI/gpt-neox-20b"
filename = "conversation.txt"

def get_model_and_tokenizer():
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.attn_config['attn_impl'] = 'triton'
    config.init_device = 'cuda:0'  # For fast initialization directly on GPU!

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer

def chatbot(text, model, tokenizer):
    # Open the file in append mode
    with open(filename, "a+") as f:
        f.seek(0)  # go to the beginning of the file
        previous_conversation = f.read().splitlines()  # read previous conversation
        f.write(text + "\n")  # save user input

        gen = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)  # device=0 for CUDA

        with torch.cuda.amp.autocast():
            generated_text = gen(text, max_length=100)

        response = generated_text[0]['generated_text']
        f.write(response + "\n")  # save chatbot response to file
        f.seek(0)  # go to the beginning of the file again
        updated_conversation = f.read().splitlines()  # read updated conversation

    return "\n".join(updated_conversation[len(previous_conversation):])

def main():
    if not os.path.exists(filename):
        open(filename, "w").close()  # create file if it doesn't exist

    model, tokenizer = get_model_and_tokenizer()

    while True:
        prompt = input("Please enter a prompt: ")
        response = chatbot(prompt, model, tokenizer)
        print("Generated Text:", response)

if __name__ == '__main__':
    main()
