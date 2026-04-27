@echo off
title LTX-Video UI
echo Starting LTX-Video...
echo.

:: Activate the virtual environment
call "%~dp0env\Scripts\activate.bat"

:: Install gradio if not already installed
python -c "import gradio" 2>nul || (
    echo Installing Gradio...
    pip install gradio
    echo.
)

:: Launch the app
python "%~dp0app.py"

:: If the app exits with an error, pause so the user can read it
if errorlevel 1 (
    echo.
    echo Something went wrong. See the error above.
    pause
)
