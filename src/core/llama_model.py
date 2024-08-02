
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from dotenv import load_dotenv
import os

load_dotenv()

LLAMA_MODEL_PATH = os.getenv('LLAMA_MODEL_PATH')

def load_llama_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    tokenizer, model = load_llama_model()
    prompt = "Hello, how can I assist you today?"
    response = generate_response(tokenizer, model, prompt)
    print(response)
