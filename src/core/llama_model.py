
import transformers
import torch
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_ID = os.getenv('LLAMA_MODEL_ID', "meta-llama/Llama-2-7b-chat-hf")

def load_llama_pipeline():
    pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL_ID,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline

def generate_response(pipeline, messages, max_new_tokens=256):
    outputs = pipeline(
        messages,
        max_new_tokens=max_new_tokens,
    )
    return outputs[0]["generated_text"]

# Example usage
if __name__ == "__main__":
    pipeline = load_llama_pipeline()
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello, who are you?"},
    ]
    response = generate_response(pipeline, messages)
    print(response)
