@echo off
:: Auther tozawa
:: Date 20180924
:: 学習，検証，テストデータを作成する

:: 引数の確認
set ARGC=0
for %%a in ( %* ) do set /a ARGC+=1
if %ARGC% neq 2 (
  echo [%~f0] [InputImg] [OutputDir]
  exit /b
)

:: Set const variables
set BASE_PATH=%~dp0
set PY_PATH=%BASE_PATH%generate_data.py
set PYTHON=%USERPROFILE%/Anaconda3/python.exe
set INPUT_IMG=%1
set OUTPUT_DIR=%2

call %PYTHON% %PY_PATH% -i %INPUT_IMG% -o %OUTPUT_DIR%
