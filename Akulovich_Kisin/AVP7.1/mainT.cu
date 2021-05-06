#include <iostream>
#include <string>
#include "mpi.h"
#include "cudaCode.h"

#define NUM_IMAGES 16

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int rank, size;
	float gpuTime, totalGpuTime, cpuTime;
	float sumGpuTime = 0, sumTotalGpuTime = 0, sumCpuTime = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	for (int i = rank; i < NUM_IMAGES; i += size) {
		if (processImage(std::to_string(i) + ".ppm", gpuTime, totalGpuTime, cpuTime)) {
			std::cout << "[" << rank << "] Processing " <<  std::to_string(i) + ".ppm finished successfully!" << std::endl;
			sumGpuTime += gpuTime;
			sumTotalGpuTime += totalGpuTime;
			sumCpuTime += cpuTime;
		}
	}

	if (rank == 0) {
		float time;

		for (int i = 1; i < size; ++i) {
			MPI_Recv(&time, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sumGpuTime += time;

			MPI_Recv(&time, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sumTotalGpuTime += time;
			
			MPI_Recv(&time, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sumCpuTime += time;
		}

		std::cout << "Average GPU Time: " << sumGpuTime / NUM_IMAGES << std::endl;
		std::cout << "Average total GPU Time: " << sumTotalGpuTime / NUM_IMAGES << std::endl;
		std::cout << "Average CPU Time: " << sumCpuTime / NUM_IMAGES << std::endl;
	}
	else {	
		MPI_Send(&sumGpuTime, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&sumTotalGpuTime, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&sumCpuTime, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}