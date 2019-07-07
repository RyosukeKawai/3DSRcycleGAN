@echo off
:: Auther tozawa
:: Date 20180921
:: MIを使って，位置合わせ

:: 引数の確認
set ARGC=0
for %%a in ( %* ) do set /a ARGC+=1
if %ARGC% neq 4 (
  echo [%~f0] [FixedImg] [MovingImg] [FixedMaskImg] [OutputDir]
  exit /b
)

:: Set const variables
set BASE_PATH=%~dp0
set EXE_PATH=%BASE_PATH%elastix-4.9.0-win64/elastix.exe
set PARA_PATH=%BASE_PATH%parameter_files/usingMI/ParameterFiles.txt
set FIXED_IMG=%1
set MOVING_IMG=%2
set FIXED_MASK_IMG=%3
set OUTPUT_DIR=%4

:: Start Registration
echo %EXE_PATH% -f %FIXED_IMG% -m %MOVING_IMG% -out %OUTPUT_DIR% -p %PARA_PATH% -fMask %FIXED_MASK_IMG%
