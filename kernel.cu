#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "inc/helper_image.h"

#include <cstdlib>
#include <cstdio>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;

const int AMOUNT_OF_THREADS_X = 32;
const int AMOUNT_OF_THREADS_Y = 16;

typedef struct pixel
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

typedef struct int_pixel
{
	int r;
	int g;
	int b;
};

#define CUDA_DEBUG

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result, const char* err)
{
#if defined(DEBUG)  || defined(CUDA_DEBUG)
	if (result != cudaSuccess)
	{
		cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << " at: " << err << endl;
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

__global__ void ApplyPrewittFilter(unsigned char* input_data, unsigned char* output_data, const int width, const int height, const int padded_width, const int padded_height, const int in_pitch, const int out_pitch)
{
	const int x = blockIdx.x * AMOUNT_OF_THREADS_X + threadIdx.x;
	const int y = blockIdx.y * AMOUNT_OF_THREADS_Y + threadIdx.y;

	const int int_widht = in_pitch / sizeof(int);
	const int output_int_width = out_pitch / sizeof(int);
	const int width_border = (width + sizeof(int) - 1) / sizeof(int);

	uchar4* reintterpreted_input = reinterpret_cast<uchar4*>(input_data);
	uchar4* reintterpreted_output = reinterpret_cast<uchar4*>(output_data);

	__shared__ uchar4 shared_memory[AMOUNT_OF_THREADS_Y + 2][AMOUNT_OF_THREADS_X + 2];
	if (x <= int_widht && y <= padded_height)
	{
		shared_memory[threadIdx.y][threadIdx.x] = reintterpreted_input[y * int_widht + x];

		if (y + AMOUNT_OF_THREADS_Y < padded_height && threadIdx.y < 2)
			shared_memory[AMOUNT_OF_THREADS_Y + threadIdx.y][threadIdx.x] = reintterpreted_input[(AMOUNT_OF_THREADS_Y + y) * int_widht + x];

		if (!(threadIdx.x % 31))
		{
			shared_memory[threadIdx.y][threadIdx.x + 1] = reintterpreted_input[y * int_widht + x + 1];
			shared_memory[threadIdx.y][threadIdx.x + 2] = reintterpreted_input[y * int_widht + x + 2];
		}

		if ((!(threadIdx.y % 14) || !(threadIdx.y % 15)) && threadIdx.x >= 30)
		{
			shared_memory[threadIdx.y + 2][threadIdx.x + 2] = reintterpreted_input[(y + 2) * int_widht + x + 2];
		}
	}

	__syncthreads();

	if (x <= int_widht && y <= padded_height) {
		uchar4 out_uchar4 = { y * output_int_width + x };
		uchar4 first_int = shared_memory[threadIdx.y][threadIdx.x];
		uchar4 second_int = shared_memory[threadIdx.y][threadIdx.x + 1];
		uchar4 third_int = shared_memory[threadIdx.y][threadIdx.x + 2];
		uchar4 fourth_int = shared_memory[threadIdx.y + 1][threadIdx.x];
		uchar4 fifth_int = shared_memory[threadIdx.y + 1][threadIdx.x + 1];
		uchar4 sixth_int = shared_memory[threadIdx.y + 1][threadIdx.x + 2];
		uchar4 seventh_int = shared_memory[threadIdx.y + 2][threadIdx.x];
		uchar4 eighth_int = shared_memory[threadIdx.y + 2][threadIdx.x + 1];
		uchar4 nineth_int = shared_memory[threadIdx.y + 2][threadIdx.x + 2];

		int tmp1, tmp2;
		/////////////////////
		tmp1 = (seventh_int.x + seventh_int.w + eighth_int.z) - (first_int.x + first_int.w + second_int.z);
		tmp2 = (second_int.z + fifth_int.z + eighth_int.z) - (first_int.x + fourth_int.x + seventh_int.x);
		if (tmp1 > 255) tmp1 = 255;
		if (tmp1 < 0) tmp1 = 0;
		if (tmp2 > 255) tmp2 = 255;
		if (tmp2 < 0) tmp2 = 0;
		out_uchar4.x = (tmp1 >= tmp2) ? unsigned char(tmp1) : unsigned char(tmp2);
		////////////////////
		tmp1 = (seventh_int.y + eighth_int.x + eighth_int.w) - (first_int.y + second_int.x + second_int.w);
		tmp2 = (second_int.w + fifth_int.w + eighth_int.w) - (first_int.y + fourth_int.y + seventh_int.y);
		if (tmp1 > 255) tmp1 = 255;
		if (tmp1 < 0) tmp1 = 0;
		if (tmp2 > 255) tmp2 = 255;
		if (tmp2 < 0) tmp2 = 0;
		out_uchar4.y = (tmp1 > tmp2) ? (unsigned char)tmp1 : (unsigned char)tmp2;
		/////////////////////////
		tmp1 = (seventh_int.z + eighth_int.y + nineth_int.x) - (first_int.z + second_int.y + third_int.x);
		tmp2 = (third_int.x + sixth_int.x + nineth_int.x) - (first_int.z + fourth_int.z + seventh_int.z);
		if (tmp1 > 255) tmp1 = 255;
		if (tmp1 < 0) tmp1 = 0;
		if (tmp2 > 255) tmp2 = 255;
		if (tmp2 < 0) tmp2 = 0;
		out_uchar4.z = (tmp1 > tmp2) ? (unsigned char)tmp1 : (unsigned char)tmp2;
		//////////////////////////
		tmp1 = (seventh_int.w + eighth_int.z + nineth_int.y) - (first_int.w + second_int.z + third_int.y);
		tmp2 = (third_int.y + sixth_int.y + nineth_int.y) - (first_int.w + fourth_int.w + seventh_int.w);
		if (tmp1 > 255) tmp1 = 255;
		if (tmp1 < 0) tmp1 = 0;
		if (tmp2 > 255) tmp2 = 255;
		if (tmp2 < 0) tmp2 = 0;
		out_uchar4.w = (tmp1 > tmp2) ? (unsigned char)tmp1 : (unsigned char)tmp2;

		if (x < output_int_width && y < height)
			reintterpreted_output[y * output_int_width + x] = out_uchar4;
	}

}

pixel* PadDataByOnePixel(pixel* input_data, int width, int height)
{
	const int new_width = width + 2;
	const int new_height = height + 2;

	pixel* output_data = new pixel[new_width * new_height];

	// copy initial part
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			output_data[(y + 1) * new_width + x + 1] = input_data[y * width + x];
		}
	}

	output_data[0] = input_data[0];
	output_data[new_width - 1] = input_data[width - 1];
	output_data[new_width * (new_height - 1)] = input_data[width * (height - 1)];
	output_data[new_width * new_height - 1] = input_data[width * height - 1];

	for (int x = 0; x < width; x++)
	{
		output_data[x + 1] = input_data[x];
		output_data[(new_height - 1) * new_width + x + 1] = input_data[width * (height - 1) + x];
	}

	for (int y = 0; y < height; y++)
	{
		output_data[(y + 1) * new_width] = input_data[y * width];
		output_data[(y + 1) * new_width + new_width - 1] = input_data[y * width + width - 1];
	}

	return output_data;
}

// Use PadDataByOneByte transforamtion for input before using this function
void PrewittFilter(pixel* input_matrix, pixel* output_matrix, const int width, const int height, const int padded_width, const int padded_height)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			pixel perfect = { 0 };
			int_pixel tmp1 = { 0 };
			int_pixel tmp2 = { 0 };

			tmp1.r = (
				(input_matrix[(y + 2) * padded_width + x].r + input_matrix[(y + 2) * padded_width + x + 1].r + input_matrix[(y + 2) * padded_width + x + 2].r)
				-
				(input_matrix[y * padded_width + x].r + input_matrix[y * padded_width + x + 1].r + input_matrix[y * padded_width + x + 2].r)
				);
			tmp1.g = (
				(input_matrix[(y + 2) * padded_width + x].g + input_matrix[(y + 2) * padded_width + x + 1].g + input_matrix[(y + 2) * padded_width + x + 2].g)
				-
				(input_matrix[y * padded_width + x].g + input_matrix[y * padded_width + x + 1].g + input_matrix[y * padded_width + x + 2].g)
				);
			tmp1.b = (
				(input_matrix[(y + 2) * padded_width + x].b + input_matrix[(y + 2) * padded_width + x + 1].b + input_matrix[(y + 2) * padded_width + x + 2].b)
				-
				(input_matrix[y * padded_width + x].b + input_matrix[y * padded_width + x + 1].b + input_matrix[y * padded_width + x + 2].b)
				);

			tmp2.r = (
				(input_matrix[y * padded_width + x + 2].r + input_matrix[(y + 1) * padded_width + x + 2].r + input_matrix[(y + 2) * padded_width + x + 2].r)
				-
				(input_matrix[y * padded_width + x].r + input_matrix[(y + 1) * padded_width + x].r + input_matrix[(y + 2) * padded_width + x].r)
				);
			tmp2.g = (
				(input_matrix[y * padded_width + x + 2].g + input_matrix[(y + 1) * padded_width + x + 2].g + input_matrix[(y + 2) * padded_width + x + 2].g)
				-
				(input_matrix[y * padded_width + x].g + input_matrix[(y + 1) * padded_width + x].g + input_matrix[(y + 2) * padded_width + x].g)
				);
			tmp2.b = (
				(input_matrix[y * padded_width + x + 2].b + input_matrix[(y + 1) * padded_width + x + 2].b + input_matrix[(y + 2) * padded_width + x + 2].b)
				-
				(input_matrix[y * padded_width + x].b + input_matrix[(y + 1) * padded_width + x].b + input_matrix[(y + 2) * padded_width + x].b)
				);

			if (tmp1.r > 255) tmp1.r = 255; if (tmp1.r < 0) tmp1.r = 0;
			if (tmp1.g > 255) tmp1.g = 255; if (tmp1.g < 0) tmp1.g = 0;
			if (tmp1.b > 255) tmp1.b = 255; if (tmp1.b < 0) tmp1.b = 0;
			if (tmp2.r > 255) tmp2.r = 255; if (tmp2.r < 0) tmp2.r = 0;
			if (tmp2.g > 255) tmp2.g = 255; if (tmp2.g < 0) tmp2.g = 0;
			if (tmp2.b > 255) tmp2.b = 255; if (tmp2.b < 0) tmp2.b = 0;

			perfect.r = (tmp1.r > tmp2.r) ? (unsigned char)tmp1.r : (unsigned char)tmp2.r;
			perfect.g = (tmp1.g > tmp2.g) ? (unsigned char)tmp1.g : (unsigned char)tmp2.g;
			perfect.b = (tmp1.b > tmp2.b) ? (unsigned char)tmp1.b : (unsigned char)tmp2.b;
			output_matrix[y * width + x] = perfect;

		}
	}
}

void ApplyPrewittFilter(pixel* input_matrix, pixel* output_matrix, const int width, const int height)
{
	pixel* padded_input_matrix = PadDataByOnePixel(input_matrix, width, height);

	const int padded_width = width + 2;
	const int padded_height = height + 2;
	//PrintPixelMatrix(padded_input_matrix, padded_width, padded_height);
	PrewittFilter(padded_input_matrix, output_matrix, width, height, padded_width, padded_height);
}

// Check errors
void postprocess(const unsigned char* in_data, const unsigned char* out_data, int width, int height, float cpu, float gpu)
{

	int cnt = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (in_data[y * width + x] != out_data[y * width + x])
			{
				//cout << endl << "*** FAILED ***" << endl;
				if (cnt < 100) printf("Error at x:%d y:%d\n", x, y);
				cnt++;
			}
		}
	}
	printf("NUM ERRORS: %d \n", cnt);
	cout << "Time difference: " << (cpu - gpu) << endl;
}

int main()
{
	char file_name[] = "nature.ppm";
	char cpu_resilt_file_name[] = "CPU_result.ppm";
	char gpu_resilt_file_name[] = "GPU_result.ppm";

	size_t width = 0;
	size_t height = 0;
	int channels = 0;

	pixel* input_data = nullptr;

	__loadPPM(
		file_name, reinterpret_cast<unsigned char**>(&input_data),
		reinterpret_cast<unsigned int*>(&width),
		reinterpret_cast<unsigned int*>(&height),
		reinterpret_cast<unsigned int*>(&channels)
	);

	cout << width << " " << height << " " << channels << endl << endl;

	const size_t padded_width = width + 2;
	const size_t padded_height = height + 2;

	const size_t size = width * height;

	const size_t width_in_bytes = width * sizeof(pixel);
	const size_t padded_width_in_bytes = padded_width * sizeof(pixel);

	const size_t size_in_bytes = width_in_bytes * height;

	pixel* cpu_output_data = new pixel[size];
	pixel* gpu_output_data = new pixel[size];

	// ********************************************************************************************************

	cout << "Filtering via CPU" << endl;
	auto start_cpu = chrono::steady_clock::now();
	ApplyPrewittFilter(input_data, cpu_output_data, width, height);
	auto end_cpu = chrono::steady_clock::now();
	auto cpu_time = end_cpu - start_cpu;
	float cpu_time_count = chrono::duration<double, milli>(cpu_time).count();
	cout << "CPU time: " << chrono::duration<double, milli>(cpu_time).count() << endl;

	// ********************************************************************************************************

	size_t input_pitch = 0;
	pixel* padded_input = PadDataByOnePixel(input_data, width, height);
	unsigned char* pitched_input_data = nullptr;

	checkCuda(cudaMallocPitch(reinterpret_cast<void**>(&pitched_input_data), &input_pitch, padded_width_in_bytes, padded_height), "CudaMallocPitch");
	checkCuda(cudaMemcpy2D(pitched_input_data, input_pitch, reinterpret_cast<unsigned char**>(padded_input), padded_width_in_bytes, padded_width_in_bytes, padded_height, cudaMemcpyHostToDevice), "CudaMemcpy2D");

	size_t output_pitch = 0;
	unsigned char* pitched_output_data = nullptr;
	checkCuda(cudaMallocPitch(reinterpret_cast<void**>(&pitched_output_data), &output_pitch, width_in_bytes, height), "CudaMallocPitch");

	float gpu_time_count = 0;
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent), "CudaEventCreate");
	checkCuda(cudaEventCreate(&stopEvent), "CudaEventCreate");

	//
	cout << "Filtering via GPU" << " pitch: " << input_pitch << " " << output_pitch << endl;

	int aligned_width = (input_pitch + AMOUNT_OF_THREADS_X - 1) / AMOUNT_OF_THREADS_X;
	int aligned_height = (height + AMOUNT_OF_THREADS_Y - 1) / AMOUNT_OF_THREADS_Y;
	dim3 dimGrid(aligned_width, aligned_height, 1);
	dim3 dimBlock(AMOUNT_OF_THREADS_X, AMOUNT_OF_THREADS_Y, 1);

	checkCuda(cudaEventRecord(startEvent, 0), "CudaEventRecord");

	ApplyPrewittFilter << <dimGrid, dimBlock >> > (pitched_input_data, pitched_output_data, width_in_bytes, height, padded_width_in_bytes, padded_height, input_pitch, output_pitch);
	checkCuda(cudaEventRecord(stopEvent, 0), "CudaEventRecord");
	checkCuda(cudaEventSynchronize(stopEvent), "CudaEventSynchronize");
	checkCuda(cudaEventElapsedTime(&gpu_time_count, startEvent, stopEvent), "CudaEventElapsedTime");
	cout << "GPU time: " << gpu_time_count << endl;

	checkCuda(cudaMemcpy2D(reinterpret_cast<unsigned char*>(gpu_output_data), width_in_bytes, pitched_output_data, output_pitch, width_in_bytes, height, cudaMemcpyDeviceToHost), "Memcpu2d");

	// ********************************************************************************************************

	// check
	postprocess(reinterpret_cast<unsigned char*>(cpu_output_data), reinterpret_cast<unsigned char*>(gpu_output_data), width * 3, height, cpu_time_count, gpu_time_count);

	__savePPM(cpu_resilt_file_name, reinterpret_cast<unsigned char*>(cpu_output_data), width, height, channels);
	__savePPM(gpu_resilt_file_name, reinterpret_cast<unsigned char*>(gpu_output_data), width, height, channels);

	checkCuda(cudaEventDestroy(startEvent), "CudaEventDestroy");
	checkCuda(cudaEventDestroy(stopEvent), "CudaEventDestroy");
	checkCuda(cudaFree(pitched_input_data), "CudaFree");
	checkCuda(cudaFree(pitched_output_data), "CudaFree");
	delete[] input_data;
	delete[] cpu_output_data;
	delete[] gpu_output_data;
}
