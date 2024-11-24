@echo off
setlocal

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set OUTPUT_DIR=..\..\lib

if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

"%CUDA_PATH%\bin\nvcc.exe" ^
    -shared ^
    -Xcompiler "/MD" ^
    -Xcompiler "/wd4098" ^
    -Xlinker "/NODEFAULTLIB:LIBCMT" ^
    -o %OUTPUT_DIR%\cudabridge.dll ^
    fixed_point.cu cuda_bridge.cu

if errorlevel 1 (
    echo Error compilant CUDA
    exit /b 1
)

echo Compilació CUDA completada amb èxit