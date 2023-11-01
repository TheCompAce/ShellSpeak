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
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  | find /V "already satisfied"
    pip install -r requirements.txt | find /V "already satisfied"
    python -m spacy download en_core_web_sm
)
call env\Scripts\activate
pip install -r requirements.txt | find /V "already satisfied"

echo Starting ShellSpeak...

python main.py /start %*