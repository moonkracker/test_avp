CXX        = /opt/cuda-7.5/bin/nvcc
LD         = $(CXX)

LIBS_PATH = -L/usr/lib 
LIBS =  -lmpi -lrt -lpthread
INCLUDE_PATH = -I/usr/lib/x86_64-linux-gnu/openmpi/include/
FLAGS = -g
TARGET = "main.cu"
OBIN = "cudampi"
ARCH = -arch=sm_20

all: $(TARGET)

$(TARGET):
	$(LD) $(INCLUDE_PATH) $(FLAGS) $(TARGET) -o $(OBIN) $(LIBS_PATH) $(LIBS)
