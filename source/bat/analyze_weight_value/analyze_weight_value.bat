@echo off
::To evaluate each iterations

set logDir=F:\project\3D-SRGAN\work\analyze_weight_value
set modelDir=F:\project\3D-SRGAN\results\training0002
set PYTHON=C:\Users\tozawa\Anaconda3\python.exe
set runPy=F:\project\3D-SRGAN\source\analyze_weight_value.py

echo 69e9612a19d31b741b9ef3cfac16edbd3562a0c8 > %logDir%/memo.txt

call :run	75000


PAUSE
exit

:run
set arg1=%1
mkdir %logDir%\%arg1%
if ERRORLEVEL 1 GOTO :skip

echo;
echo MakeDirectory %logDir%\%arg1%

::Analysis
PYTHON %runPy%  -m %modelDir%\dis_iter_%arg1%.npz -o %logDir%\%arg1%

:skip
exit /b
