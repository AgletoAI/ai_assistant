import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import venv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("setup.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        logger.error(f"Error executing command: {command}")
        logger.error(stderr.decode())
        return False
    return True

def setup_project(project_root: str, force: bool = False):
    project_root = os.path.abspath(os.path.normpath(project_root))
    
    if os.path.exists(project_root):
        if force:
            logger.warning(f"Directory {project_root} already exists. Removing...")
            os.system(f"rmdir /S /Q {project_root}")
        else:
            logger.error(f"Directory {project_root} already exists. Use --force to overwrite.")
            sys.exit(1)

    os.makedirs(project_root, exist_ok=True)
    os.chdir(project_root)

    env_name = "ai_assistant"

    logger.info("Setting up with venv...")
    venv.create(env_name, with_pip=True)
    
    if sys.platform == "win32":
        python_path = os.path.join(env_name, "Scripts", "python.exe")
        pip_path = os.path.join(env_name, "Scripts", "pip.exe")
    else:
        python_path = os.path.join(env_name, "bin", "python")
        pip_path = os.path.join(env_name, "bin", "pip")
    
    logger.info("Upgrading pip...")
    if not run_command(f'"{python_path}" -m pip install --upgrade pip'):
        logger.error("Failed to upgrade pip")
        sys.exit(1)
    
    logger.info("Installing PyTorch...")
    pytorch_version = "2.4.0+cpu"
    torchvision_version = "0.19.0+cpu"
    torchaudio_version = "2.4.0+cpu"  # Updated version
    
    if not run_command(f'"{pip_path}" install torch=={pytorch_version} torchvision=={torchvision_version} torchaudio=={torchaudio_version} --index-url https://download.pytorch.org/whl/cpu'):
        logger.error("Failed to install PyTorch")
        sys.exit(1)
    
    logger.info("Installing additional packages...")
    if not run_command(f'"{pip_path}" install transformers==4.29.2 python-dotenv'):
        logger.error("Failed to install additional packages")
        sys.exit(1)

    # Create project files
    Path("src").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    with open("src/main.py", "w") as f:
        f.write(main_py_content)

    with open("src/llama_model.py", "w") as f:
        f.write(llama_model_py_content)

    with open(".env.example", "w") as f:
        f.write(env_example_content)

    with open("run.py", "w") as f:
        f.write(run_script_content)

    logger.info("\nProject setup complete!")
    if sys.platform == "win32":
        logger.info(f"To activate the environment: {env_name}\\Scripts\\activate.bat")
    else:
        logger.info(f"To activate the environment: source {env_name}/bin/activate")
    logger.info("To run the AI assistant: python run.py")

main_py_content = '''
import os
import sys
import logging
from dotenv import load_dotenv
from llama_model import load_llama_pipeline, generate_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("logs/assistant.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info("Python path:")
        for path in sys.path:
            logger.info(f"  {path}")
        
        load_dotenv()

        model, tokenizer = load_llama_pipeline()
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Hello, who are you?"},
        ]
        response = generate_response(model, tokenizer, messages)
        logger.info(f"AI Response: {response}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
'''

llama_model_py_content = '''
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
import logging
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

load_dotenv()
MODEL_ID = os.getenv('LLAMA_MODEL_ID', "meta-llama/Meta-Llama-3.1-8B-Instruct")

def load_llama_pipeline() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading Llama pipeline: {e}", exc_info=True)
        raise

def generate_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                      messages: List[Dict[str, str]], max_new_tokens: int = 256) -> str:
    try:
        prompt = format_messages(messages)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
        return tokenizer.decode(outputs[0])
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise

def format_messages(messages: List[Dict[str, str]]) -> str:
    formatted = ""
    for message in messages:
        if message["role"] == "system":
            formatted += f"System: {message['content']}\\n"
        elif message["role"] == "user":
            formatted += f"Human: {message['content']}\\n"
        elif message["role"] == "assistant":
            formatted += f"Assistant: {message['content']}\\n"
    formatted += "Assistant: "
    return formatted
'''

env_example_content = '''
LLAMA_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
DEBUG_MODE=False
'''

run_script_content = '''
import os
import sys
import subprocess

def main():
    venv_dir = 'ai_assistant'
    python_executable = os.path.join(os.getcwd(), venv_dir, "Scripts", "python.exe")
    main_script = os.path.join("src", "main.py")
    
    try:
        command = f'"{python_executable}" "{main_script}"'
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the AI assistant: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

def main():
    parser = argparse.ArgumentParser(description="Set up AI Assistant project")
    parser.add_argument("project_name", help="Name of the project directory")
    parser.add_argument("--force", action="store_true", help="Force overwrite if project directory already exists")
    args = parser.parse_args()

    try:
        setup_project(args.project_name, args.force)
    except Exception as e:
        logger.error(f"An unexpected error occurred during project setup: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
