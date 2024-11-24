@echo off
setlocal

REM Configura el PATH per incloure les llibreries necessàries
set "PATH=%cd%\lib;%CUDA_PATH%\bin;%PATH%"

REM Executa el programa
.\bin\ld_sylls.exe