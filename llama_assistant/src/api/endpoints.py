
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate-command")
def generate_command(task_description: str):
    # Logic to generate command based on task description
    return {"command": f"Generated command for {task_description}"}
