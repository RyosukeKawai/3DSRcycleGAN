@echo off
set /P N=いくつまわしますか？
for /L %%i in (1,1,%N%) do (
start %1
)
