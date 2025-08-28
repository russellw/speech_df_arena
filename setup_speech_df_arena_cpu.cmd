@echo off
setlocal

REM ===== Fast + stable: conda-forge only, then pip for the rest =====
REM Versions chosen to satisfy librosa==0.9.2 + numpy==1.23.3, with SciPy pinned to 1.10.1.

set ENV_NAME=speech-df-arena
if not "%~1"=="" set ENV_NAME=%~1

echo.
echo === Nuking any half-created env (safe if it doesn't exist) ===
call conda deactivate
call conda env remove -n "%ENV_NAME%" -y >NUL 2>&1

echo.
echo === Clean indexes to speed up solve ===
call conda clean -a -y >NUL 2>&1

echo.
echo === Create env in ONE shot from conda-forge only (classic solver) ===
REM --override-channels + -c conda-forge keeps it to one repo = faster, fewer conflicts
REM scipy pinned to 1.10.1 to match numpy 1.23.x + Py 3.10
call conda create -y -n "%ENV_NAME%" --override-channels -c conda-forge --solver classic ^
  python=3.10 ^
  numpy==1.23.3 ^
  scipy==1.10.1 ^
  librosa==0.9.2 ^
  pandas==2.2.3 ^
  pyyaml==6.0.2 ^
  rich==13.9.4 ^
  tqdm==4.67.1 ^
  einops==0.8.1 ^
  pytorch-lightning==2.3.2
if errorlevel 1 goto :conda_error

echo.
echo === Activate env ===
call conda activate "%ENV_NAME%"
if errorlevel 1 goto :conda_error

echo.
echo === Upgrade pip and wheel (faster installs) ===
python -m pip install --upgrade pip wheel
if errorlevel 1 goto :pip_error

echo.
echo === Install CPU PyTorch + Torchaudio via pip ===
REM No GPU; pip wheels are straightforward on Windows CPU.
pip install torch torchaudio
if errorlevel 1 goto :pip_error

echo.
echo === Install HF + pip-first libraries ===
pip install transformers==4.45.2 fairseq==1.0.0a0 speechbrain==1.0.0
if errorlevel 1 goto :pip_error

echo.
echo === Verify ===
python -V
pip --version
conda list | findstr /R /C:"^python " /C:"^numpy " /C:"^scipy " /C:"^librosa " /C:"^pandas " /C:"^pytorch-lightning " /C:"^pyyaml " /C:"^rich " /C:"^tqdm " /C:"^einops "
pip show torch torchaudio transformers fairseq speechbrain | findstr /B "Name Version"

echo.
echo === Done. Activate later with: conda activate %ENV_NAME% ===
goto :eof

:conda_error
echo [ERROR] Conda step failed. See messages above.
exit /b 1

:pip_error
echo [ERROR] Pip step failed. See messages above.
exit /b 1
