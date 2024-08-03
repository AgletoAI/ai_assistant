from src.models.llama_model import OptimizedLlamaModel

model = OptimizedLlamaModel()

def generate_command(task_description):
    prompt = f"Generate a system command to perform the following task: {task_description}"
    response = model(prompt)
    return parse_command(response)

def parse_command(response):
    # Implement parsing logic to extract command from model's response
    return response.strip()

def validate_command(command):
    # Implement command validation logic
    unsafe_keywords = ['rm', 'del', 'format', 'mkfs']
    return not any(keyword in command for keyword in unsafe_keywords)
