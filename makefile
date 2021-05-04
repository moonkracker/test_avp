CUDA_CC=/opt/cuda-7.5/bin/nvcc
MPI_CC=mpic++
CUDA_FLAGS=-arch=sm_20 -std=c++11
OUTPUT_FILE=lab7_avp
INPUT_CUDA=kernel.cu
INPUT_MPI=kernel.cu

all: cuda

cuda: $(INPUT_CUDA)
        $(CUDA_CC) $(CUDA_FLAGS) $(INPUT_CUDA) -o $(OUTPUT_FILE)
mpi: $(INPUT_MPI)
        $(MPI_CC) 


