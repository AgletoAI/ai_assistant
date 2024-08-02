@echo off
setlocal enabledelayedexpansion

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed or not in the system PATH.
    echo Please install Python and add it to your system PATH.
    pause
    exit /b 1
)

:: Set the project name and model path
set "PROJECT_NAME=llama_assistant"
set "MODEL_PATH=C:\Users\harry\meta-llama\Meta-Llama-3.1-8B-Instruct"

:: Check if setup_llama_project.py exists in the current directory
if not exist "setup_llama_project.py" (
    echo setup_llama_project.py not found in the current directory.
    echo Please ensure the script is in the same directory as this batch file.
    pause
    exit /b 1
)

:: Run the setup script
echo Running Llama 3.1 project setup...
python setup_llama_project.py "%PROJECT_NAME%" "%MODEL_PATH%" --force

if %errorlevel% neq 0 (
    echo An error occurred during the setup process.
    echo Please check the setup.log file for more information.
    pause
    exit /b 1
)

:: If successful, provide instructions
echo.
echo Setup completed successfully!
echo To activate the virtual environment, run:
echo %PROJECT_NAME%\Scripts\activate.bat
echo.
echo To run the AI assistant, navigate to the project directory and run:
echo python run.py
echo.

pause