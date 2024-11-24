# Makefile per a la creació de la llibreria de treball amb CUDA.
# CreatedAt: 2024/11/24 dg. JIQ

# Configuració de paths per Windows
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
LIB_DIR = ..\..\lib
NVCC = "$(CUDA_PATH)\bin\nvcc.exe"
LIB_NAME = cudabridge.dll

# Flags de compilació per Windows
NVCC_FLAGS = -shared -Xcompiler "/MD"

# Fonts
SOURCES = fixed_point.cu cuda_bridge.cu

# Objectiu principal
all: $(LIB_DIR)\$(LIB_NAME)

# Crea el directori lib si no existeix
$(LIB_DIR):
    if not exist $(LIB_DIR) mkdir $(LIB_DIR)

# Compila la llibreria
$(LIB_DIR)\$(LIB_NAME): $(SOURCES)
    if not exist $(LIB_DIR) mkdir $(LIB_DIR)
    $(NVCC) $(NVCC_FLAGS) -o $(LIB_DIR)\$(LIB_NAME) $(SOURCES)

clean:
    if exist $(LIB_DIR)\$(LIB_NAME) del /Q /F $(LIB_DIR)\$(LIB_NAME)