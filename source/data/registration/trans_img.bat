@echo off
:: Auther tozawa
:: Date 20180921
:: パラメータファイルを用いて，変換

:: 引数の確認
set ARGC=0
for %%a in ( %* ) do set /a ARGC+=1
if %ARGC% neq 3 (
  echo [%~f0] [InputImg] [ParameterFiles] [OutputDir]
  exit /b
)

:: Set const variables
set BASE_PATH=%~dp0
set EXE_PATH=%BASE_PATH%elastix-4.9.0-win64/transformix.exe
set INPUT_IMG=%1
set PARA_PATH=%2
set OUTPUT_DIR=%3

:: Start Transformation
call %EXE_PATH% -in %INPUT_IMG% -out %OUTPUT_DIR% -tp %PARA_PATH%
