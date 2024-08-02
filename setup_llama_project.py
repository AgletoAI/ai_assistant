import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import venv
import shutil

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

def setup_project(project_root: str, model_path: str, force: bool = False):
    project_root = os.path.abspath(os.path.normpath(project_root))
    
    if os.path.exists(project_root):
        if force:
            logger.warning(f"Directory {project_root} already exists. Removing...")
            shutil.rmtree(project_root)
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
    
    logger.info("Installing PyTorch with CUDA support...")
    pytorch_version = "2.0.1"
    cuda_version = "118"  # This corresponds to CUDA 11.8
    if not run_command(f'"{pip_path}" install torch=={pytorch_version}+cu{cuda_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_version}'):
        logger.error("Failed to install PyTorch")
        sys.exit(1)
    
    logger.info("Installing additional packages...")
    if not run_command(f'"{pip_path}" install transformers==4.29.2 python-dotenv sentencepiece'):
        logger.error("Failed to install additional packages")
        sys.exit(1)

    logger.info("Installing tokenizers...")
    if not run_command(f'"{pip_path}" install --only-binary=:all: tokenizers'):
        logger.error("Failed to install tokenizers")
        sys.exit(1)

    logger.info("Installing CUDA dependencies...")
    if not run_command(f'"{pip_path}" install pycuda'):
        logger.error("Failed to install CUDA dependencies")
        sys.exit(1)

    logger.info("Cloning and building llama.cpp...")
    if not run_command(f'git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -S . -B build -DGGML_CUDA=ON && cmake --build build --config Release'):
        logger.error("Failed to clone and build llama.cpp")
        sys.exit(1)

    logger.info("Converting model to GGUF format...")
    if not run_command(f'"{python_path}" llama.cpp/convert.py {model_path} --outfile model.gguf'):
        logger.error("Failed to convert model")
        sys.exit(1)

    converted_model_path = "model.gguf"
    quantized_model_path = "model-q4_0.gguf"

    logger.info("Quantizing model...")
    if not run_command(f'llama.cpp/build/bin/quantize {converted_model_path} {quantized_model_path} q4_0'):
        logger.error("Failed to quantize model")
        sys.exit(1)

    # Create project files
    Path("src").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    with open("src/main.py", "w") as f:
        f.write(f'''
from llama_model import LlamaModel
import os

def main():
    model_path = "{quantized_model_path}"
    model = LlamaModel(model_path)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        response = model.generate(user_input)
        print("AI:", response)

if __name__ == "__main__":
    main()
''')

    with open("src/llama_model.py", "w") as f:
        f.write('''
import subprocess
import os

class LlamaModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.llama_cpp_path = os.path.join('llama.cpp', 'build', 'bin', 'main')

    def generate(self, prompt, max_tokens=128):
        command = f'{self.llama_cpp_path} -m {self.model_path} -n {max_tokens} -p "{prompt}" --instruct'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Error running llama.cpp: {stderr.decode()}")
        return stdout.decode()
''')

    with open(".env.example", "w") as f:
        f.write(f'''
# Example environment variables
MODEL_PATH={quantized_model_path}
''')

    with open("run.py", "w") as f:
        f.write('''
import os
import sys
import dotenv

if __name__ == "__main__":
    dotenv.load_dotenv()
    
    # Add the src directory to the Python path
    src_path = os.path.join(os.path.dirname(__file__), "src")
    sys.path.append(src_path)
    
    from main import main
    main()
''')

    logger.info("\nProject setup complete!")
    if sys.platform == "win32":
        logger.info(f"To activate the environment: {env_name}\\Scripts\\activate.bat")
    else:
        logger.info(f"To activate the environment: source {env_name}/bin/activate")
    logger.info("To run the AI assistant: python run.py")

def main():
    parser = argparse.ArgumentParser(description="Set up AI Assistant project")
    parser.add_argument("project_name", help="Name of the project directory")
    parser.add_argument("model_path", help="Path to the downloaded Llama 3.1 model directory")
    parser.add_argument("--force", action="store_true", help="Force overwrite if project directory already exists")
    args = parser.parse_args()

    try:
        setup_project(args.project_name, args.model_path, args.force)
    except Exception as e:
        logger.error(f"An unexpected error occurred during project setup: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
