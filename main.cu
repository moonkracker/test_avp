#include "kernel.cuh"

using namespace std;

int main(int argc, char** argv) {
	int rank, size;

	MPI_Init (&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);//current process
	MPI_Comm_size(MPI_COMM_WORLD, &size);//number of process
	
	const char* primaryImagePath[] = {
		"avp_logo.ppm", 
		"belka.ppm", 
		"cat.ppm", 
		"nature.ppm", 
		"fire.ppm", 
		"graffiti.ppm", 
		"nvidia.ppm"};
	const char* outputImagePathCPU[] = {
		"avp_logo_CPU.ppm", 
		"belka_CPU.ppm", 
		"cat_CPU.ppm", 
		"nature_CPU.ppm", 
		"fire_CPU.ppm", 
		"graffiti_CPU.ppm", 
		"nvidia_CPU.ppm"};
	const char* outputImagePathGPU[] = {
		"avp_logo_GPU.ppm", 
		"belka_GPU.ppm", 
		"cat_GPU.ppm", 
		"nature_GPU.ppm", 
		"fire_GPU.ppm", 
		"graffiti_GPU.ppm", 
		"nvidia_GPU.ppm"};

	size_t width = 0;
	size_t height = 0;
	int channels = 0;

	pixel* input_data = nullptr;
	
	for (int i=0;i<size;i++)
		if(i == rank)
		{
			__loadPPM(primaryImagePath[i], reinterpret_cast<unsigned char**>(&input_data),
			reinterpret_cast<unsigned int*>(&width),
			reinterpret_cast<unsigned int*>(&height),
			reinterpret_cast<unsigned int*>(&channels));
		}


	const size_t padded_width = width + 2;
	const size_t padded_height = height + 2;

	const size_t size = width * height;

	const size_t width_in_bytes = width * sizeof(pixel);
	const size_t padded_width_in_bytes = padded_width * sizeof(pixel);
	
	pixel* cpu_output_data = new pixel[size];
	pixel* gpu_output_data = new pixel[size];

	// *********************************CPU**********************************************************

	cout << "Filtering via CPU" << endl;
	auto start_cpu = chrono::steady_clock::now();
	ApplyPrewittFilter(input_data, cpu_output_data, width, height);
	auto end_cpu = chrono::steady_clock::now();
	auto cpu_time = end_cpu - start_cpu;
	float cpu_time_count = chrono::duration<double, milli>(cpu_time).count();
	cout << "CPU time: " << chrono::duration<double, milli>(cpu_time).count() << endl;

	// *********************************CPU_END*******************************************************************
	
	cuda_filter(width, height, width_in_bytes, padded_width_in_bytes, input_data);

	for (int i=0;i<size;i++)
		if(i == rank)
		{
			__savePPM(outputImagePathCPU[i], reinterpret_cast<unsigned char*>(cpu_output_data), width, height, channels);
			__savePPM(outputImagePathGPU[i], reinterpret_cast<unsigned char*>(gpu_output_data), width, height, channels);
		}

	delete[] input_data;
	delete[] cpu_output_data;
	delete[] gpu_output_data;
	
	MPI_Finalize();
	return 0;
}
