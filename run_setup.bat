@echo off
cd /d C:\HarryAITime\ai_assistant
python setup_ai_assistant.py llama3.1_local --force
cd /d C:\HarryAITime\ai_assistant\llama3.1_local
echo.
echo Setup process completed.
echo If the setup was successful, you can now activate the environment and run the AI assistant.
echo Please check the setup.log file for instructions on how to activate the environment.
echo After activating the environment, you can run the AI assistant with: python run.py
echo.
echo Otherwise, you can close this window or run other commands.
cmd /k