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
