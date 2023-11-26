@echo off
if not exist env (
    echo Creating virtual environment...
    python -m venv env
    echo.

    echo Activating virtual environment...
    call env\Scripts\activate

    echo Updating pip...
    python -m pip install --upgrade pip

    call env\Scripts\deactivate

    echo Activating virtual environment...
    call env\Scripts\activate

    echo Installing dependencies...
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  | find /V "already satisfied"
    pip install -r requirements.txt | find /V "already satisfied"
    REM pip install -c torch -c nvidia faiss-gpu
    python -m spacy download en_core_web_sm
    
)
call env\Scripts\activate
pip install -r requirements.txt | find /V "already satisfied"

echo Starting ShellSpeak...

python main.py %*