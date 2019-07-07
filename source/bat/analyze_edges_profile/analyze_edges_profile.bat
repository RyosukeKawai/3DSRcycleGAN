@echo off

set PYTHON=C:\Users\tozawa\Anaconda3\python.exe
set runPy=F:\project\3D-SRGAN\source\analyze_edges_profile.py
set INPUTIMAGE="F:\experiment_data\data\validation\normalization\z-score\denoising\MicroCT.mhd"
set OUTPUT=F:\project\3D-SRGAN\work\analyze_edges_profile\20180911\coronal\normalization\denoising\MicroCT.png

set MODE=coronal
set SLICE=79
set FIXED=318
set START=118
set END=138
set DIRECTION=horizontal

PYTHON %runPy% -i %INPUTIMAGE% -o %OUTPUT% --mode %MODE% --slice %SLICE% --fixed_pos %FIXED% --start %START% --end %END% --direction %DIRECTION%
