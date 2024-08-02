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

def setup_project(project_root: str, force: bool = False):
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
    if not run_command(f'"{pip_path}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'):
        logger.error("Failed to install PyTorch")
        sys.exit(1)

    logger.info("Installing Hugging Face Transformers...")
    if not run_command(f'"{pip_path}" install transformers'):
        logger.error("Failed to install Transformers")
        sys.exit(1)

    logger.info("Installing Accelerate...")
    if not run_command(f'"{pip_path}" install accelerate'):
        logger.error("Failed to install Accelerate")
        sys.exit(1)

    logger.info("Installing additional packages...")
    packages = [
        "python-dotenv",
        "fastapi",
        "uvicorn"
    ]
    for package in packages:
        if not run_command(f'"{pip_path}" install {package}'):
            logger.error(f"Failed to install {package}")
            sys.exit(1)

    # Create project files
    Path("src").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    with open("src/llama_model.py", "w") as f:
        f.write('''
import transformers
import torch

class LlamaModel:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True
            },
            device_map="auto",
        )

    def generate(self, prompt, max_tokens=256):
        messages = [
            {"role": "user", "content": prompt},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_tokens,
        )
        return outputs[0]["generated_text"]
''')

    with open("src/api_server.py", "w") as f:
        f.write('''
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from llama_model import LlamaModel

app = FastAPI()

model = LlamaModel()

class Prompt(BaseModel):
    prompt: str

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.post("/generate")
async def generate_text(prompt: Prompt):
    try:
        response = model.generate(prompt.prompt)
        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
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
    
    from api_server import app
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
''')

    logger.info("\nProject setup complete!")
    logger.info(f"To activate the environment: {env_name}\\Scripts\\activate.bat")
    logger.info("To run the API server: python run.py")

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
