#include "Header.h"

//arch=gencode=compute=52

#define BLOCK_SIZE_X 1024
#define BLOCK_SIZE_Y 1

using namespace std;

int main(int argc, char** argv) {
	int rank, size;
	int ImageWidth = 0;
	int ImageHeight = 0;

	MPI_Init (&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);//current process
	MPI_Comm_size(MPI_COMM_WORLD, &size);//number of process

	const char primaryImagePath0[] = "supercar.ppm";
	const char outputImageCpuPath0[] = "supercarCPU0.ppm";
	const char outputImageGpuPath0[] = "supercarGPU0.ppm";
	const char outputImageGpuSharedPath0[] = "supercarGPUshared0.ppm";

	const char primaryImagePath1[] = "city.ppm";
	const char outputImageCpuPath1[] = "cityCPU1.ppm";
	const char outputImageGpuPath1[] = "cityGPU1.ppm";
	const char outputImageGpuSharedPath1[] = "cityGPUshared1.ppm";

	const char primaryImagePath2[] = "road.ppm";
	const char outputImageCpuPath2[] = "roadCPU2.ppm";
	const char outputImageGpuPath2[] = "roadGPU2.ppm";
	const char outputImageGpuSharedPath2[] = "roadGPUshared2.ppm";

	const char primaryImagePath3[] = "nature.ppm";
	const char outputImageCpuPath3[] = "natureCPU3.ppm";
	const char outputImageGpuPath3[] = "natureGPU3.ppm";
	const char outputImageGpuSharedPath3[] = "natureGPUshared3.ppm";

	const char primaryImagePath4[] = "car.ppm";
	const char outputImageCpuPath4[] = "carCPU4.ppm";
	const char outputImageGpuPath4[] = "carGPU4.ppm";
	const char outputImageGpuSharedPath4[] = "carGPUshared4.ppm";


	unsigned char* primaryImage = NULL;

	unsigned int imageWidth = 0, imageHeight = 0, channels = 0;
	
	switch (rank) {
	case 0:
		__loadPPM(primaryImagePath0, &primaryImage, &imageWidth, &imageHeight, &channels);
		ImageHeight = imageHeight;
		ImageWidth = imageWidth;
		break;
	case 1:
		__loadPPM(primaryImagePath1, &primaryImage, &imageWidth, &imageHeight, &channels);
		ImageHeight = imageHeight;
		ImageWidth = imageWidth;
		break;
	case 2:
		__loadPPM(primaryImagePath2, &primaryImage, &imageWidth, &imageHeight, &channels);
		ImageHeight = imageHeight;
		ImageWidth = imageWidth;
		break;
	case 3:
		__loadPPM(primaryImagePath3, &primaryImage, &imageWidth, &imageHeight, &channels);
		ImageHeight = imageHeight;
		ImageWidth = imageWidth;
		break;
	case 4:
		__loadPPM(primaryImagePath4, &primaryImage, &imageWidth, &imageHeight, &channels);
		ImageHeight = imageHeight;
		ImageWidth = imageWidth;
		break;
	default:
		return 0;
		break;
	}

	unsigned char* outputImageCpu = (unsigned char*)malloc(imageWidth * imageHeight * sizeof(char) * 3);
	unsigned char* outputImageGpu = (unsigned char*)malloc(imageWidth * imageHeight * sizeof(char) * 3);
	unsigned char* outputImageGpuShared = (unsigned char*)malloc(imageWidth * imageHeight * sizeof(char) * 3);
	unsigned char* resizedImage = (unsigned char*)malloc((imageWidth + 2) * (imageHeight + 2) * sizeof(char) * 3);

	resizeImage(primaryImage, resizedImage, imageWidth, imageHeight);

	cpu_filterImage(resizedImage, outputImageCpu, imageWidth, imageHeight);
	cuda_filterImage(resizedImage, outputImageGpu, false);
	cuda_filterImage(resizedImage, outputImageGpuShared, true);
	
	//cout << "Start compare" << endl;

	/*if (checkEquality(outputImageCpu, outputImageGpu, ImageWidth
, ImageHeight)
		&& checkEquality(outputImageGpu, outputImageGpuShared, ImageWidth
	, ImageHeight)) {
		cout << "Results are equals!" << endl;
	}
	else {
		cout << "Results are NOT equals!" << endl;
	}*/

	//cout << "Saving..." << endl;
	switch (rank) {
	case 0:
		__savePPM(outputImageCpuPath0, outputImageCpu, imageWidth, imageHeight, channels);
		
		break;
	case 1:
		__savePPM(outputImageCpuPath1, outputImageCpu, imageWidth, imageHeight, channels);
		
		break;
	case 2:
		__savePPM(outputImageCpuPath2, outputImageCpu, imageWidth, imageHeight, channels);
		
		break;
	case 3:
		__savePPM(outputImageCpuPath3, outputImageCpu, imageWidth, imageHeight, channels);
		
		break;
	case 4:
		__savePPM(outputImageCpuPath4, outputImageCpu, imageWidth, imageHeight, channels);
		
		break;
	default:
		return 0;
		break;
	}
	cout << "Saved" << endl;

	free(primaryImage);
	free(resizedImage);
	free(outputImageCpu);
	
	MPI_Finalize();
	return 0;
}

