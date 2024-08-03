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

:: Set the project name
set "PROJECT_NAME=llama_assistant"

:: Check if setup_llama_project.py exists in the current directory
if not exist "setup_llama_project.py" (
    echo setup_llama_project.py not found in the current directory.
    echo Please ensure the script is in the same directory as this batch file.
    pause
    exit /b 1
)

:: Run the setup script
echo Running Llama 3.1 project setup...
python setup_llama_project.py "%PROJECT_NAME%" --force

if %errorlevel% neq 0 (
    echo An error occurred during the setup process.
    echo Please check the setup.log file for more information.
    pause
    exit /b 1
)

:: If successful, activate the virtual environment and run the AI assistant
echo.
echo Setup completed successfully!
echo Activating virtual environment and running the AI assistant...
call %PROJECT_NAME%\ai_assistant\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Change to the project directory
cd %PROJECT_NAME%

:: Run the AI assistant
python main.py

if %errorlevel% neq 0 (
    echo An error occurred while running the AI assistant.
    pause
    exit /b 1
)

:: Deactivate the virtual environment (this line will only be reached if the user manually stops the server)
call deactivate

pause