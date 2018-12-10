
#include "GaussianLUSingleBlock.h"

const int Dimensions = 3;
const int Radon = Dimensions + 2;
const int M = Radon;

const int threadsX = 32;
const int threadsY = 32;

__global__ void gaussianLUKernel(float* devU, float* devL, float* devP/*, const int M*/) {

	//Tell one thread to load float devGaussianMat into a shared memory.
	__shared__ float sharedU[M][M];
	__shared__ float sharedL[M][M];
	__shared__ float sharedP[M][M];
	__shared__ float switchRow[M];

	int maxIndex =0;
	int maxVal = 0;


	int i = 0;
	int j = 0;
	int k = 0;
	int t = 0;
	int x = threadIdx.x;
	int y = threadIdx.y;
	float temp;

	if (y < M && x < M) {
		sharedU[y][x] = *(devU + y * M + x);
		if (y == x)
			sharedL[y][x] = sharedP[y][x] = 1;
		else
			sharedL[y][x] = sharedP[y][x] = 0;
	}

	__syncthreads();

	//printf("Block Dim %d\t%d", blockDim.x, blockDim.y);

	if (M > blockDim.y || M > blockDim.x)
		//CHANGE THIS
		printf("Dimensions are too large\n");
	else {

		for (k = 0; k < M-1; k++) {

			if (y == 1 && x == 1) {
				for (i = 0; i < M*M; i++) {
					if (i % M == 0)
						printf("\n");
					printf("%.2f\t", sharedU[i / M][i % M]);
				}
				printf("\n\n\n");
				for (i = 0; i < M*M; i++) {
					if (i % M == 0)
						printf("\n");
					printf("%.2f\t", sharedL[i / M][i % M]);
				}
				printf("\n\n\n");
				for (i = 0; i < M*M; i++) {
					if (i % M == 0)
						printf("\n");
					printf("%.2f\t", sharedP[i / M][i % M]);
				}
				printf("\n\n\n");
			}
			/*if (y == 1 && x == 1) {

				/*for (t = 0; t < width*height; t++) {
					if (t % width == 0)
						printf("\n");
					printf("%.2f\t", *(devGaussianMat + t));
					printf("\n\n");
				}

				for (t = 0; t < width*height; t++) {
					if (t % width == 0)
						printf("\n");
					printf("%.2f\t", sharedMat[t / width][t % width]);
				}
				printf("\n\n");
			}*/
			maxVal = 0;

			for (i = k; i < M; i++) {
				if (abs(sharedU[i][k] > maxVal))
					maxIndex = i;
			}
			__syncthreads();
			if(maxIndex == k)
				continue;
			else {
				if ((y == 0) && (x >= k) && (x < M)) {
					switchRow[x] = sharedU[maxIndex][x];
					sharedU[maxIndex][x] = sharedU[k][x];
					sharedU[k][x] = switchRow[x];
				}
				__syncthreads();
				if ((y == 0) && (x >= 0) && (x < k-1)) {
					switchRow[x] = sharedL[maxIndex][x];
					sharedL[maxIndex][x] = sharedL[k][x];
					sharedL[k][x] = switchRow[x];
				}
				__syncthreads();
				if ((y == 0) && (x >= 0) && (x < M)) {
					switchRow[x] = sharedP[maxIndex][x];
					sharedP[maxIndex][x] = sharedP[k][x];
					sharedP[k][x] = switchRow[x];
				}
				__syncthreads();
			}

			if ((y >= k + 1) && (y < M) && (x == 0))
				sharedL[y][k] = sharedU[y][k] / sharedU[k][k];
			__syncthreads();
			if ((y >= k + 1) && (y < M) && (x >= k) && (x < M))
				sharedU[y][x] = sharedU[y][x] - (sharedL[y][k] * sharedU[k][x]);
		
			/*for (k = 0; k < height; k++) {

				if (k == i)
					continue;
				temp = *(devGaussianMat + k * width + i);
				//printf("%d\n", k*width);
				//printf("temp - %.2f\n", temp);
				for (j = 0; j < width; j++) {
					//printf("k-%d,j-%d val-%.2f\n", k, j, *(devGaussianMat + k * width + j));
					//printf("temp - %.2f\n", temp);
					*(devGaussianMat + k * width + j) = *(devGaussianMat + k * width + j) - (temp * *(devGaussianMat + i * width + j));
					//printf("k-%d,j-%d val-%.2f\n", k,j, *(devGaussianMat + k * width + j));
				}
			}*/
			/*for (t = 0; t < width*height; t++) {
				if (t % width == 0)
					printf("\n");
				printf("%.2f\t", *(devGaussianMat + t));
			}*/
		}

		/*if (y == 1 && x == 1) {
			for (t = 0; t < width*height; t++) {
				if (t % width == 0)
					printf("\n");
				printf("%.2f\t", sharedMat[t / width][t % width]);
			}
			printf("\n\n");
		}*/
		if (y < M && x < M) {
			*(devU + y * M + x) = sharedU[y][x];
			*(devL + y * M + x) = sharedL[y][x];
			*(devP + y * M + x) = sharedP[y][x];
		}

	}

	/*if (y == 1 && x == 1) {
		for (t = 0; t < width*height; t++) {
			if (t % width == 0)
				printf("\n");
			printf("%.5f\t", *(devGaussianMat + t));
		}
		printf("\n\n");
	}*/

}
int singleLUBlock()
{

	//TODO
	//Functionality to read a hypothesis, convert to matrix for guassian elimination. Get Dimensions
	//Allocate memory for 2-D arrays

	int width = Radon;
	int height = Dimensions + 1;
	float r;
	cudaEvent_t start, stop;

	float* hostU = (float *)malloc(M * M * sizeof(float));
	float* hostL = (float *)malloc(M * M * sizeof(float));
	float* hostP = (float *)malloc(M * M * sizeof(float));

	float *devU;
	float *devL;
	float *devP;

	cudaMalloc(&(devU), M * M * sizeof(float));
	cudaMalloc(&(devL), M * M * sizeof(float));
	cudaMalloc(&(devP), M * M * sizeof(float));

	dim3 grid(1, 1, 1);
	dim3 block(threadsY, threadsX, 1);
	int i = 0;

	//For now we'll just fill matrices with dummy values

	for (i = 0; i < M*M; i++)
		*(hostU + i) = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 10000.0));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMemcpy(devU, hostU, M * M * sizeof(float), cudaMemcpyHostToDevice);

	gaussianLUKernel << <grid, block >> > (devU, devL, devP/*, Radon*/);

	cudaMemcpy(hostU, devU, M * M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostL, devL, M * M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostP, devP, M * M * sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < M*M; i++) {
		if (i % M == 0)
			printf("\n");
		printf("%.2f\t", *(hostU + i));
	}
	printf("\n\n\n");
	for (i = 0; i < M*M; i++) {
		if (i % M == 0)
			printf("\n");
		printf("%.2f\t", *(hostL + i));
	}
	printf("\n\n\n");
	for (i = 0; i < M*M; i++) {
		if (i % M == 0)
			printf("\n");
		printf("%.2f\t", *(hostP + i));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);

	printf("Time %.2f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(devU);
	cudaFree(devL);
	cudaFree(devP);

	free(hostU);
	free(hostL);
	free(hostP);
	return 0;
}
