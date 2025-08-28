@echo off
setlocal enabledelayedexpansion

REM ================================================
REM Speech DF Arena environment (Windows, CPU-only)
REM - Python 3.10 (compatible with librosa 0.9.2 & numpy 1.23.3)
REM - Conda for most packages
REM - Pip for transformers / fairseq / speechbrain
REM ================================================

REM Usage: setup_speech_df_arena_cpu.cmd [ENV_NAME]
REM Example: setup_speech_df_arena_cpu.cmd speech-df-arena
set ENV_NAME=%1
if "%ENV_NAME%"=="" set ENV_NAME=speech-df-arena

echo.
echo === Creating conda env "%ENV_NAME%" with Python 3.10 ===
call conda create -y -n "%ENV_NAME%" python=3.10
if errorlevel 1 goto :conda_error

echo.
echo === Activating env ===
call conda activate "%ENV_NAME%"
if errorlevel 1 goto :conda_error

echo.
echo === Installing core packages with conda-forge ===
call conda install -y -c conda-forge ^
  numpy==1.23.3 ^
  pandas==2.2.3 ^
  scipy==1.15.1 ^
  librosa==0.9.2 ^
  pytorch-lightning==2.3.2 ^
  pyyaml==6.0.2 ^
  rich==13.9.4 ^
  tqdm==4.67.1 ^
  einops==0.8.1
if errorlevel 1 goto :conda_error

echo.
echo === Installing PyTorch + Torchaudio (CPU) from pytorch channel ===
call conda install -y -c pytorch pytorch torchaudio cpuonly
if errorlevel 1 goto :conda_error

echo.
echo === Upgrading pip ===
python -m pip install --upgrade pip
if errorlevel 1 goto :pip_error

echo.
echo === Installing pip-first packages (transformers, fairseq, speechbrain) ===
pip install ^
  transformers==4.45.2 ^
  fairseq==1.0.0a0 ^
  speechbrain==1.0.0
if errorlevel 1 goto :pip_error

echo.
echo === Done. Environment "%ENV_NAME%" is ready. ===
python -V
pip --version
echo.
echo --- Key packages installed ---
conda list | findstr /R /C:"^numpy " /C:"^pandas " /C:"^scipy " /C:"^librosa " /C:"^pytorch " /C:"^torchaudio " /C:"^pytorch-lightning " /C:"^pyyaml " /C:"^rich " /C:"^tqdm " /C:"^einops "
pip show transformers | findstr /B "Name Version"
pip show fairseq     | findstr /B "Name Version"
pip show speechbrain | findstr /B "Name Version"
echo.

goto :eof

:conda_error
echo.
echo [ERROR] A conda step failed. Make sure 'conda' is available in CMD and try again.
exit /b 1

:pip_error
echo.
echo [ERROR] A pip step failed. See the error log above.
exit /b 1
