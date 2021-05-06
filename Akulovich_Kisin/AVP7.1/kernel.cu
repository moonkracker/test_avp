#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_image.h"

#include <iostream> 
#include <cuda_runtime.h> 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <tuple>

#pragma comment(lib, "cudart") 

#define BLOCK_SIZE_X 1024
#define BLOCK_SIZE_Y 1
#define IMAGE_WIDTH 8000
#define IMAGE_HEIGHT 4000
//typedef BYTE char;

using namespace std;

void cpu_filterImage(unsigned char*,unsigned  char*, int, int);
void cuda_filterImage(unsigned char*,unsigned  char*, bool);
void cudaCheckStatus(cudaError_t);
void resizeImage(unsigned char*,unsigned char*, int, int);
bool checkEquality(unsigned char*,unsigned  char*, int, int);
__global__ void cuda_filterImage(unsigned char*,unsigned  char*, size_t, size_t);
__global__ void cuda_filterImageShared(unsigned char*, unsigned char*, size_t, size_t);
__device__ char sumPixels(char, char, char, char, char);
__device__ short pack(uchar3);
__device__ uchar2 unpack(short);

//int main() {
//	
//}

void cuda_filterImage(unsigned char* inMatrix, unsigned char* outMatrix, bool optimizationFlag) {
	float resultTime;

	unsigned char* device_inMatrix;
	unsigned char* device_outMatrix;

	cudaEvent_t cuda_startTime;
	cudaEvent_t cuda_endTime;

	cudaCheckStatus(cudaEventCreate(&cuda_startTime));
	cudaCheckStatus(cudaEventCreate(&cuda_endTime));

	int numOfBlocksInRow = (int)ceil((double)(IMAGE_WIDTH) / (BLOCK_SIZE_X * 2));
	int numOfBlockInColumn = IMAGE_HEIGHT;
	int blocksNeeded = numOfBlockInColumn * numOfBlocksInRow;



	size_t pitchInMatrix = 0, pitchOutMatrix = 0;
	int gridSizeY = numOfBlockInColumn;
	int gridSizeX = numOfBlocksInRow;

	cudaCheckStatus(cudaMallocPitch((void**)&device_inMatrix, &pitchInMatrix, (IMAGE_WIDTH + 2) * 3, gridSizeY + 2));
	cudaCheckStatus(cudaMallocPitch((void**)&device_outMatrix, &pitchOutMatrix, IMAGE_WIDTH * 3, gridSizeY));
	cudaCheckStatus(cudaMemcpy2D(
		device_inMatrix, pitchInMatrix,
		inMatrix, (IMAGE_WIDTH + 2) * 3,
		(IMAGE_WIDTH + 2) * 3, gridSizeY + 2,
		cudaMemcpyHostToDevice));

	dim3 blockSize(BLOCK_SIZE_X);
	dim3 gridSize(gridSizeX, gridSizeY);

	cudaCheckStatus(cudaEventRecord(cuda_startTime, NULL));

	if (optimizationFlag) {
		cuda_filterImageShared << < gridSize, blockSize >> > (device_inMatrix, device_outMatrix, pitchInMatrix, pitchOutMatrix);
	}
	else {
		cuda_filterImage << < gridSize, blockSize >> > (device_inMatrix, device_outMatrix, pitchInMatrix, pitchOutMatrix);
	}

	cudaCheckStatus(cudaPeekAtLastError());
	cudaCheckStatus(cudaEventRecord(cuda_endTime, NULL));
	cudaCheckStatus(cudaEventSynchronize(cuda_endTime));

	cudaCheckStatus(cudaEventElapsedTime(&resultTime, cuda_startTime, cuda_endTime));

	if (optimizationFlag) {
		printf(" CUDA time with optimization: %lf seconds\n", (double)resultTime / CLOCKS_PER_SEC);
	}
	else {
		printf(" CUDA time: %lf seconds\n", (double)resultTime / CLOCKS_PER_SEC);
	}

	cudaCheckStatus(cudaMemcpy2D(
		outMatrix, IMAGE_WIDTH * 3,
		device_outMatrix, pitchOutMatrix,
		IMAGE_WIDTH * 3, gridSizeY,
		cudaMemcpyDeviceToHost)
	);

	inMatrix = &inMatrix[(IMAGE_WIDTH + 2) * gridSizeY * 3];
	outMatrix = &outMatrix[IMAGE_WIDTH * gridSizeY * 3];

	cudaCheckStatus(cudaFree(device_inMatrix));
	cudaCheckStatus(cudaFree(device_outMatrix));

}

void cpu_filterImage(unsigned char* primaryImage,unsigned  char* outputImage, int imageWidth, int imageHeight) {
	primaryImage = &primaryImage[(imageWidth + 2 + 1) * 3];

	clock_t startTime, endTime;
	startTime = clock();
	//0  1  0
	//1 -4  1
	//0  1  0
	for (auto i = 0; i < imageHeight; i++) {
		for (auto j = 0; j < imageWidth; j++) {
			for (auto k = 0; k < 3; k++) {     //rgb
				short sum = 0;

				sum += -4 * primaryImage[(i * (imageWidth + 2) + j) * 3 + k];
				sum += primaryImage[(i * (imageWidth + 2) + j + 1) * 3 + k];
				sum += primaryImage[(i * (imageWidth + 2) + j - 1) * 3 + k];

				sum += primaryImage[((i + 1) * (imageWidth + 2) + j) * 3 + k];

				sum += primaryImage[((i - 1) * (imageWidth + 2) + j) * 3 + k];

				if (sum > 255) {
					sum = 255;
				}
				else if (sum < 0) {
					sum = 0;
				}

				outputImage[(i * imageWidth + j) * 3 + k] = (char)sum;
			}
		}
	}
	endTime = clock();
	printf(" CPU time: %lf seconds\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
}

__global__ void cuda_filterImage(unsigned char* inMatrix, unsigned char* outMatrix, size_t pitchInMatrix, size_t pitchOutMatrix) {
	int remainderElements = (IMAGE_WIDTH % (blockDim.x * 2)) / 2;

	if (remainderElements != 0 && (blockIdx.x + 1) % gridDim.x == 0 && threadIdx.x >= remainderElements) {
		return;
	}

	short* startOfProcessingRow = (short*)&inMatrix[pitchInMatrix * blockIdx.y + blockIdx.x * blockDim.x * 2 * 3 + threadIdx.x * 2 * 3];

	short a2 = startOfProcessingRow[1];
	short a3 = startOfProcessingRow[2];
	short a4 = startOfProcessingRow[3];
	short a5 = startOfProcessingRow[4];

	short b1 = startOfProcessingRow[pitchInMatrix / 2];
	short b2 = startOfProcessingRow[pitchInMatrix / 2 + 1];
	short b3 = startOfProcessingRow[pitchInMatrix / 2 + 2];
	short b4 = startOfProcessingRow[pitchInMatrix / 2 + 3];
	short b5 = startOfProcessingRow[pitchInMatrix / 2 + 4];
	short b6 = startOfProcessingRow[pitchInMatrix / 2 + 5];

	short c2 = startOfProcessingRow[pitchInMatrix + 1];
	short c3 = startOfProcessingRow[pitchInMatrix + 2];
	short c4 = startOfProcessingRow[pitchInMatrix + 3];
	short c5 = startOfProcessingRow[pitchInMatrix + 4];

	uchar2 aa2 = unpack(a2);
	uchar2 aa3 = unpack(a3);
	uchar2 aa4 = unpack(a4);
	uchar2 aa5 = unpack(a5);

	uchar2 bb1 = unpack(b1);
	uchar2 bb2 = unpack(b2);
	uchar2 bb3 = unpack(b3);
	uchar2 bb4 = unpack(b4);
	uchar2 bb5 = unpack(b5);
	uchar2 bb6 = unpack(b6);

	uchar2 cc2 = unpack(c2);
	uchar2 cc3 = unpack(c3);
	uchar2 cc4 = unpack(c4);
	uchar2 cc5 = unpack(c5);

	uchar3 firstPixel, secondPixel;
	firstPixel.x = sumPixels(aa2.y, bb1.x, bb2.y, bb4.x, cc2.y);
	firstPixel.y = sumPixels(aa3.x, bb1.y, bb3.x, bb4.y, cc3.x);
	firstPixel.z = sumPixels(aa3.y, bb2.x, bb3.y, bb5.x, cc3.y);
	secondPixel.x = sumPixels(aa4.x, bb2.y, bb4.x, bb5.y, cc4.x);
	secondPixel.y = sumPixels(aa4.y, bb3.x, bb4.y, bb6.x, cc4.y);
	secondPixel.z = sumPixels(aa5.x, bb3.y, bb5.x, bb6.y, cc5.x);

	outMatrix = &outMatrix[blockIdx.y * pitchOutMatrix + threadIdx.x * 2 * 3 + blockIdx.x * blockDim.x * 2 * 3];
	outMatrix[0] = firstPixel.x;
	outMatrix[1] = firstPixel.y;
	outMatrix[2] = firstPixel.z;
	outMatrix[3] = secondPixel.x;
	outMatrix[4] = secondPixel.y;
	outMatrix[5] = secondPixel.z;
}

__global__ void cuda_filterImageShared(unsigned char* inMatrix,unsigned  char* outMatrix, size_t pitchInMatrix, size_t pitchOutMatrix) {
	int remainderElements = (IMAGE_WIDTH % (blockDim.x * 2)) / 2;

	if (remainderElements != 0 && (blockIdx.x + 1) % gridDim.x == 0 && threadIdx.x >= remainderElements) {
		return;
	}

	__shared__ short sharedMemoryIn[3][(BLOCK_SIZE_X + 1) * 3];
	__shared__ short sharedMemoryOut[BLOCK_SIZE_X * 3];

	short* startOfProcessingRow = (short*)&inMatrix[blockIdx.y * pitchInMatrix + blockIdx.x * blockDim.x * 2 * 3];
	short* outputRow = (short*)&outMatrix[blockIdx.y * pitchOutMatrix + blockIdx.x * blockDim.x * 2 * 3];

	if (threadIdx.x == 0) {
		short* tempPointer = &startOfProcessingRow[threadIdx.x];

		sharedMemoryIn[0][threadIdx.x] = tempPointer[0];
		sharedMemoryIn[0][threadIdx.x + 1] = tempPointer[1];
		sharedMemoryIn[0][threadIdx.x + 2] = tempPointer[2];

		sharedMemoryIn[1][threadIdx.x] = tempPointer[pitchInMatrix / 2];
		sharedMemoryIn[1][threadIdx.x + 1] = tempPointer[pitchInMatrix / 2 + 1];
		sharedMemoryIn[1][threadIdx.x + 2] = tempPointer[pitchInMatrix / 2 + 2];

		sharedMemoryIn[2][threadIdx.x] = tempPointer[pitchInMatrix];
		sharedMemoryIn[2][threadIdx.x + 1] = tempPointer[pitchInMatrix + 1];
		sharedMemoryIn[2][threadIdx.x + 2] = tempPointer[pitchInMatrix + 2];
	}

	startOfProcessingRow = &startOfProcessingRow[(threadIdx.x + 1) * 3];

	sharedMemoryIn[0][threadIdx.x * 3 + 3] = startOfProcessingRow[0];
	sharedMemoryIn[0][threadIdx.x * 3 + 4] = startOfProcessingRow[1];
	sharedMemoryIn[0][threadIdx.x * 3 + 5] = startOfProcessingRow[2];

	sharedMemoryIn[1][threadIdx.x * 3 + 3] = startOfProcessingRow[pitchInMatrix / 2];
	sharedMemoryIn[1][threadIdx.x * 3 + 4] = startOfProcessingRow[pitchInMatrix / 2 + 1];
	sharedMemoryIn[1][threadIdx.x * 3 + 5] = startOfProcessingRow[pitchInMatrix / 2 + 2];

	sharedMemoryIn[2][threadIdx.x * 3 + 3] = startOfProcessingRow[pitchInMatrix];
	sharedMemoryIn[2][threadIdx.x * 3 + 4] = startOfProcessingRow[pitchInMatrix + 1];
	sharedMemoryIn[2][threadIdx.x * 3 + 5] = startOfProcessingRow[pitchInMatrix + 2];


	__syncthreads();

	short a2 = sharedMemoryIn[0][threadIdx.x * 3 + 1];
	short a3 = sharedMemoryIn[0][threadIdx.x * 3 + 2];
	short a4 = sharedMemoryIn[0][threadIdx.x * 3 + 3];
	short a5 = sharedMemoryIn[0][threadIdx.x * 3 + 4];

	short b1 = sharedMemoryIn[1][threadIdx.x * 3];
	short b2 = sharedMemoryIn[1][threadIdx.x * 3 + 1];
	short b3 = sharedMemoryIn[1][threadIdx.x * 3 + 2];
	short b4 = sharedMemoryIn[1][threadIdx.x * 3 + 3];
	short b5 = sharedMemoryIn[1][threadIdx.x * 3 + 4];
	short b6 = sharedMemoryIn[1][threadIdx.x * 3 + 5];

	short c2 = sharedMemoryIn[2][threadIdx.x * 3 + 1];
	short c3 = sharedMemoryIn[2][threadIdx.x * 3 + 2];
	short c4 = sharedMemoryIn[2][threadIdx.x * 3 + 3];
	short c5 = sharedMemoryIn[2][threadIdx.x * 3 + 4];

	uchar2 aa2 = unpack(a2);
	uchar2 aa3 = unpack(a3);
	uchar2 aa4 = unpack(a4);
	uchar2 aa5 = unpack(a5);

	uchar2 bb1 = unpack(b1);
	uchar2 bb2 = unpack(b2);
	uchar2 bb3 = unpack(b3);
	uchar2 bb4 = unpack(b4);
	uchar2 bb5 = unpack(b5);
	uchar2 bb6 = unpack(b6);

	uchar2 cc2 = unpack(c2);
	uchar2 cc3 = unpack(c3);
	uchar2 cc4 = unpack(c4);
	uchar2 cc5 = unpack(c5);

	uchar3 firstPixel, secondPixel;
	firstPixel.x = sumPixels(aa2.y, bb1.x, bb2.y, bb4.x, cc2.y);
	firstPixel.y = sumPixels(aa3.x, bb1.y, bb3.x, bb4.y, cc3.x);
	firstPixel.z = sumPixels(aa3.y, bb2.x, bb3.y, bb5.x, cc3.y);
	secondPixel.x = sumPixels(aa4.x, bb2.y, bb4.x, bb5.y, cc4.x);
	secondPixel.y = sumPixels(aa4.y, bb3.x, bb4.y, bb6.x, cc4.y);
	secondPixel.z = sumPixels(aa5.x, bb3.y, bb5.x, bb6.y, cc5.x);

	short* tempSharedOut = &sharedMemoryOut[threadIdx.x * 3];

	tempSharedOut[0] = ((firstPixel.y << 8) | firstPixel.x);
	tempSharedOut[1] = ((secondPixel.x << 8) | firstPixel.z);
	tempSharedOut[2] = ((secondPixel.z << 8) | secondPixel.y);

	outputRow = &outputRow[threadIdx.x * 3];

	outputRow[0] = tempSharedOut[0];
	outputRow[1] = tempSharedOut[1];
	outputRow[2] = tempSharedOut[2];
}

__device__ short pack(uchar3 pixelLine)
{
	return (pixelLine.y << 8) | pixelLine.x;
}

__device__ uchar2 unpack(short c)
{
	uchar2 pixelLine;
	pixelLine.x = (char)(c);
	pixelLine.y = (char)(c >> 8);

	return pixelLine;
}

__device__ char sumPixels(char a2, char a4, char a5, char a6, char a8)
{
	int32_t result = 0;

	result = a2 + a4 + -4 * a5 + a6 + a8;

	if (result > 255)
		result = 255;
	else if (result < 0)
		result = 0;

	return (char)result;
}

void resizeImage(unsigned char* primaryImage,unsigned char* resizedImage, int imageWidth, int imageHeight) {
	//int n=0;
	for (int i = 0,n=0; i < imageHeight; i++, n++) {
		for (int j = 0, m = 0; j < imageWidth; j++, m++) {
			resizedImage[(n * (imageWidth + 2) + m) * 3] = primaryImage[(i * imageWidth + j) * 3];
			resizedImage[(n * (imageWidth + 2) + m) * 3 + 1] = primaryImage[(i * imageWidth + j) * 3 + 1];
			resizedImage[(n * (imageWidth + 2) + m) * 3 + 2] = primaryImage[(i * imageWidth + j) * 3 + 2];

			if (j == 0 || j == imageWidth - 1) {
				m++;
				resizedImage[(n * (imageWidth + 2) + m) * 3] = primaryImage[(i * imageWidth + j) * 3];
				resizedImage[(n * (imageWidth + 2) + m) * 3 + 1] = primaryImage[(i * imageWidth + j) * 3 + 1];
				resizedImage[(n * (imageWidth + 2) + m) * 3 + 2] = primaryImage[(i * imageWidth + j) * 3 + 2];
			}
		}

		if (n == 0 || n == imageHeight) {
			i--;
		}
	}
}

void cudaCheckStatus(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		cout << "CUDA return error code: " << cudaStatus;
		cout << " " << cudaGetErrorString(cudaStatus) << endl;
		exit(-1);
	}
}

bool checkEquality(unsigned char* firstImage, unsigned char* secondImage, int imageWidth, int imageHeight) {
	for (int i = 0; i < imageWidth * imageHeight; i++) {
		if (fabs(firstImage[i] - secondImage[i]) > 1) {
			return false;
		}
	}
	return true;
}
