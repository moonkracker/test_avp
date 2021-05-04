CUDA_CC=/opt/cuda-7.5/bin/nvcc
MPI_CC=mpic++
CUDA_FLAGS=-arch=sm_20 -std=c++11
OUTPUT_FILE=lab7_avp
INPUT_CUDA=kernel.cu

all: cuda

cuda: $(INPUT_FILE)
	$(CUDA_CC) $(CUDA_FLAGS) $(INPUT_CUDA) -o $(OUTPUT_FILE)

