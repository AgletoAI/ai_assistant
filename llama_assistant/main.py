
import uvicorn
from multiprocessing import Process
from src.api.endpoints import app
from src.collaborative.interface import launch_collaborative_interface
from src.config.settings import API_HOST, API_PORT

def run_fastapi():
    uvicorn.run(app, host=API_HOST, port=API_PORT)

def run_collaborative_interface():
    launch_collaborative_interface()

if __name__ == "__main__":
    # Start FastAPI server in a separate process
    fastapi_process = Process(target=run_fastapi)
    fastapi_process.start()

    # Run collaborative interface in the main process
    run_collaborative_interface()

    # Wait for FastAPI process to finish
    fastapi_process.join()
