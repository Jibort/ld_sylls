@echo off
setlocal

REM Configura CUDA_PATH si no està definit
if "%CUDA_PATH%"=="" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
)

REM Comprova si CUDA_PATH existeix
if not exist "%CUDA_PATH%" (
    echo Error: CUDA no trobat a %CUDA_PATH%
    echo Si us plau, instal·la CUDA o ajusta CUDA_PATH
    exit /b 1
)

REM Crea directoris si no existeixen
if not exist "lib" mkdir lib
if not exist "bin" mkdir bin

echo Compilant codi CUDA...
cd internal\cuda
call build_cuda.bat
if errorlevel 1 (
    echo Error compilant CUDA
    exit /b 1
)
cd ..\..

REM Configura variables per CGO
set "CGO_LDFLAGS=-L%cd%\lib -lcudabridge -L%CUDA_PATH%\lib\x64 -lcudart"
set "CGO_CFLAGS=-I%CUDA_PATH%\include"
set "PATH=%cd%\lib;%CUDA_PATH%\bin;%PATH%"

echo Compilant projecte Go...
go build -o bin\ld_sylls.exe .\cmd\ld_sylls

if errorlevel 1 (
    echo Error compilant Go
    exit /b 1
)

echo Build completat amb èxit!
echo Per executar el programa: .\bin\ld_sylls.exe