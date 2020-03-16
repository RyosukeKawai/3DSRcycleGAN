@echo off
::To evaluate each iterations

:: Set const variables
set A=%~dp0
set A=%A:~0,-2%
for %%A in (%A%) do set A=%%~dpA
set A=%A:~0,-2%
for %%A in (%A%) do set A=%%~dpA
set A=%A:~0,-2%
for %%A in (%A%) do set A=%%~dpA

set BASE_PATH=%A%
set PYTHON=%USERPROFILE%/Anaconda3/python.exe
set INFERENCE_PY=%BASE_PATH%inference.py
set A=%~dp0
set SUM_LOG_PY=%A%summarize_results.py

::======================================================
:: Variables that you should change
set LOAD_MODEL_DIR=%BASE_PATH%results\training
set DATA_DIR=G:\data
set OUTPUT_DIR=results\inference\PLR_HR_SN
set FILENAME=val_fn
=======
::======================================================
set LOG_DIR=%BASE_PATH%%OUTPUT_DIR%

if exist %LOG_DIR%/README.txt (goto :no_make_file) else goto :make_file

:make_file
echo LOAD_MODEL_DIR: %LOAD_MODEL_DIR% >> %LOG_DIR%/README.txt
echo DATA_DIR: %DATA_DIR% >> %LOG_DIR%/README.txt
echo OUTPUT_DIR: %OUTPUT_DIR% >> %LOG_DIR%/README.txt

:no_make_file
for /l %%i in (10000,1000,100000) do call :run	%%i
=======


PAUSE
call PYTHON SUM_LOG_PY -R %LOG_DIR%
exit


:run
set arg1=%1
mkdir %LOG_DIR%\%arg1%
if ERRORLEVEL 1 GOTO :skip

echo;
echo MakeDirectory %LOG_DIR%\%arg1%

::Inference
PYTHON %INFERENCE_PY% -g 0 -m %LOAD_MODEL_DIR%\gen_iter_%arg1%.npz -o %OUTPUT_DIR%\%arg1% -R %DATA_DIR%

:skip
exit /b
