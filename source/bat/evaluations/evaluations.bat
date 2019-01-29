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
<<<<<<< HEAD
set LOAD_MODEL_DIR=%BASE_PATH%results\pretraining0006
set DATA_DIR=F:\experiment_data\96
set OUTPUT_DIR=results\inference0009
=======
set LOAD_MODEL_DIR=%BASE_PATH%results\training0010
set DATA_DIR=F:\experiment_data\data
set OUTPUT_DIR=results\inference0017\test
>>>>>>> MI-exp
::======================================================
set LOG_DIR=%BASE_PATH%%OUTPUT_DIR%

if exist %LOG_DIR%/README.txt (goto :no_make_file) else goto :make_file

:make_file
echo LOAD_MODEL_DIR: %LOAD_MODEL_DIR% >> %LOG_DIR%/README.txt
echo DATA_DIR: %DATA_DIR% >> %LOG_DIR%/README.txt
echo OUTPUT_DIR: %OUTPUT_DIR% >> %LOG_DIR%/README.txt

:no_make_file
<<<<<<< HEAD
call :run	1000
call :run	5000
call :run	10000
call :run	15000
call :run	20000
call :run	25000
call :run	30000
call :run	35000
call :run	40000
call :run	45000
call :run	50000
call :run	55000
call :run	60000
call :run	65000
call :run	70000
call :run	75000
call :run	80000
call :run	85000
call :run	90000
call :run	95000
call :run	100000
=======
call :run	81000
>>>>>>> MI-exp

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
