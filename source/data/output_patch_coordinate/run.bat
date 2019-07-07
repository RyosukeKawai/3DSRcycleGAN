@echo off
:: Auther tozawa
:: Date 20181011
:: パッチに分割する

:: 引数の確認
set ARGC=0
for %%a in ( %* ) do set /a ARGC+=1
if %ARGC% neq 3 (
  echo [%~f0] [InputImg] [MaskImg] [OutputFileDir]
  exit /b
)

:: Set const variables
set BASE_PATH=%~dp0
set A=%~dp0
set A=%A:~0,-2%
for %%A in (%A%) do set A=%%~dpA
set EXE_PATH=%A%x64/Release/output_patch_coordinate.exe
set INPUT_IMG=%1
set MASK_IMG=%2
set OUTPUT_FILE_DIR=%3
set PATCH_SIDE=64
set SAMPLING_INTERVAL=32

call %EXE_PATH% %INPUT_IMG% %MASK_IMG% %OUTPUT_FILE_DIR% %PATCH_SIDE% %SAMPLING_INTERVAL%
