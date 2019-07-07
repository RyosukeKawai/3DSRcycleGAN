@echo off
set /P N=How many programs do you want to run?
for /L %%i in (1,1,%N%) do (
start %1
)
