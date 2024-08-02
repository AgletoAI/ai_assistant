@echo off
echo Setting up AI Assistant...

REM Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
pip install -r requirements.txt

REM Copy .env.example to .env if it doesn't exist
if not exist .env (
    copy .env.example .env
    echo Created .env file. Please update it with your settings.
)

echo Installation complete!
echo Please update the .env file with your specific settings before running the assistant.