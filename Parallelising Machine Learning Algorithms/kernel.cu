
#include "kernel.h"

#define	DIMENSIONS 3
#define RADON 5

__global__ void gaussianKernel(float* devGaussianMat, int height, int width) {
	

	int i = 0;
	int j = 0;
	int k = 0;
	int t = 0;
	float temp;

	for (t = 0; t < width*height; t++) {
		if (t % width == 0)
			printf("\n");
		printf("%.2f\t", *(devGaussianMat + t));
	}
	//printf("\n\n");

	for (i = 0; i < height; i++) {
		if (*(devGaussianMat + i * width + i) != 1) {
			temp = *(devGaussianMat + i * width + i);
			for (j = 0; j < width; j++) {
				*(devGaussianMat + i * width + j) = *(devGaussianMat + i * width + j) / temp;
			}
		}
		for (k = i + 1; k < height; k++) {
			temp = *(devGaussianMat + k * width + i);
			//printf("%d\n", k*width);
			//printf("temp - %.2f\n", temp);
			for (j = 0; j < width; j++) {
				//printf("k-%d,j-%d val-%.2f\n", k, j, *(devGaussianMat + k * width + j));
				//printf("temp - %.2f\n", temp);
				*(devGaussianMat + k * width + j) = *(devGaussianMat + k * width + j) - (temp * *(devGaussianMat + i * width + j));
				//printf("k-%d,j-%d val-%.2f\n", k,j, *(devGaussianMat + k * width + j));
			}
		}
		for (t = 0; t < width*height; t++) {
			if (t % width == 0)
				printf("\n");
			printf("%.2f\t", *(devGaussianMat + t));
		}
	}

}
int main()
{
	//TODO
	//Functionality to read a hypothesis, convert to matrix for guassian elimination. Get Dimensions
	//Allocate memory for 2-D arrays

	int width = RADON;
	int height = DIMENSIONS + 1;

	float* hostGaussianMat = (float *)malloc(width * height * sizeof(float));

	float* devGaussianMat;

	cudaMalloc(&(devGaussianMat), height * width * sizeof(float));

	dim3 grid(1, 1, 1);
	dim3 block(1, 1, 1);
	int i = 0;

		//For now we'll just fill matrices with dummy values

	for(i = 0; i < width*height; i++)
		*(hostGaussianMat + i) = (i*i) - (2 * i) + 4;

	cudaMemcpy(devGaussianMat, hostGaussianMat, height * width * sizeof(float), cudaMemcpyHostToDevice);

	gaussianKernel<<<grid,block>>>(devGaussianMat, height, width);

	cudaMemcpy(hostGaussianMat, devGaussianMat, height * width * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(devGaussianMat);

	for (i = 0; i < width*height; i++) {
		if (i % width == 0)
			printf("\n");
		printf("%.2f\t",*(hostGaussianMat + i));
	}

	free(hostGaussianMat);
	return 0;
}
