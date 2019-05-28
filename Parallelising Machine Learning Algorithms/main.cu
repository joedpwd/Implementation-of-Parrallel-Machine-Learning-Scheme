#include "main.h"

int main(int argc, char **argv) {

	//Used for Iteration
	int i=0;
	int j=0;
	int k=0;

	//Get the dimensions of the data
	//Set Valuues
	//h will be a hyper parameter passed to the program.

	const int d = 2;

	const int r = d + 2; //Assuming d = 2

	const int h = 3; //Hyper parameter

	const int m = d + 1; //Equivalent to d + 1

	const int rh = pow(r, h);

	//Perform rh instances of ML problem

	//Size of Data is r^h * d, where d is the no of features
	double *data = (double *)malloc(sizeof(double) * rh * d);


	//Create a vector of threads, one thread per execution of radon machine operation.
	//std::thread *thArray = (std::thread *)malloc(sizeof(std::thread) * pow(r, h - 1));
	std::vector<std::thread> thVect;

	//Read data in from CSV, data is stored in long long type and casted back into double type.
	std::ifstream dataFile;
	std::string t;
	std::string::size_type sz;
	dataFile.open("C:/Users/jxd45/Documents/Python Scripts/csvtest.csv");
	long long *test = (long long *)malloc(sizeof(long long));
	if (dataFile.is_open())
	{
		while (std::getline(dataFile, t))
		{
			std::cout << t << '\n';
			sz = 0;
			for (j = 0; j < d; j++) {
				*test = std::stoll(t.substr(sz), &sz);
				sz++;
				*(data + (i++)) = *reinterpret_cast<double *>(test);
			}
		}
		dataFile.close();
	}

	else
	{
		std::cout << "Unable to open file";

		return 0;
	}

	for (i = 0; i < rh * d; i++) {
		printf("%.10f\n", *(data + i));
	}


	//Iterations of radon tree
	for (i = 0; i < h; i++) {
		for (j = 0; j < pow(r, h-1 - i); j++) {
			thVect.push_back(std::thread(radonInstance, (data + (d*j*r)), d));
			//radonInstance((data + (d*j*r)), d);
		}
		for (std::thread & th : thVect)
		{
			// If thread Object is Joinable then Join that thread.
			if (th.joinable())
				th.join();
		}
		thVect.clear();
		for (j = 0; j < pow(r, h-1 - i); j++) {
			for(k=0;k<d;k++)
				*(data + (j*d) + k) = *(data + (r*j*d) + k);
		}
	}

	free(test);
	free(data);
}

void radonInstance(double *dataPoints, const int d) {

	int i, j;

	const int r = d + 2; //Assuming d = 2

	const int m = d + 1;

	double *hostA = (double *)(malloc(m*m * sizeof(double)));
	double *hostB = (double *)(malloc(m * sizeof(double)));
	double *hostX = (double *)(malloc(m * sizeof(double)));
	double *hostLU = (double *)(malloc(m*m * sizeof(double)));
	int *hostIpiv = (int *)(malloc(m * sizeof(int)));
	int *hostInfo = (int *)(malloc(sizeof(int)));

	long long *Acopy;
	long long *Bcopy;
	long long *Xcopy;

	//Format raw data into correct format for LU factorisation.
	for (j = 0; j < d; j++) {
		hostB[j] = -*(dataPoints+j);
	}

	hostB[j] = -1;

	for (i = 1; i < r; i++) {
		for (j = 0; j < d; j++) {
			hostA[(i - 1)*m + j] = *(dataPoints + (i*d) + j);
		}
		hostA[(i - 1)*m + j] = 1;
	}

	//Perform LU Factorisation
	denseLUSolver(hostA, hostB, hostX, hostLU, hostIpiv, hostInfo, m);

	/*for (i = 0; i < m; i++)
		printf("X\t%d\t%.4f\n", i, *(hostX+i));*/

	double lambda = 1;
	double *hypothesis = (double *)malloc(sizeof(double)*d);

	//Obtain correct hypothesis for instance. We have chosen to obtain the hypothesis by
	//using the positive index set I. L0 is always 1 (positive) and is therefore always in this set.
	//Therefore the first loop is to copy the first data points for this value into the 
	//hypothesis array. This array is the sum of data points whose index corresponds to the
	//values in the positive index set I.

	for (i = 0; i < d; i++)
		hypothesis[i] = *(dataPoints + i);

	for (i = 0; i < m; i++) {
		if (*(hostX + i) >= 0) {
			lambda += *(hostX + i);
			for (j = 0; j < d; j++)
				hypothesis[j] += *(hostX + i) * *(dataPoints + ((i + 1) *d) + j);
		}
		else
			continue;
	}

	printf("RESULT\n");

	/*for (i = 0; i < d; i++) {
		printf("%.5f\n", hypothesis[i]);
	}*/
	//printf("%.5f\n", lambda);
	for (i = 0; i < d; i++) {
		hypothesis[i] /= lambda;
		//printf("%.5f\n", lambda);
		printf("%.5f\n", hypothesis[i]);
		*(dataPoints + i) = hypothesis[i];
	}

	Acopy = reinterpret_cast<long long *>(hostA);
	Bcopy = reinterpret_cast<long long *>(hostB);
	Xcopy = reinterpret_cast<long long *>(hostX);

	/*std::ofstream A, B, X;
	A.open("C:/Users/jxd45/Documents/Python Scripts/A.csv", std::ios::out | std::ios::app);
	B.open("C:/Users/jxd45/Documents/Python Scripts/B.csv", std::ios::out | std::ios::app);
	X.open("C:/Users/jxd45/Documents/Python Scripts/X.csv", std::ios::out | std::ios::app);
	if (A.fail() | B.fail() | X.fail()) {
		std::cout << "Unable to open file";
	}
	else {
		for (i = 0; i < m; i++) {
			for (j = 0; j < m-1; j++) {
				A << *(Acopy + i * m + j) << ",";
			}
			A << *(Acopy + i * m + j) << std::endl;
			B << *(Bcopy + i) << std::endl;
			X << *(Xcopy + i) << std::endl;
		}
	}*/

	free(hostA);
	free(hostB);
	free(hostX);
	free(hostLU);
	free(hostIpiv);
	free(hostInfo);
	free(hypothesis);
}

/*double lambda = 0;
	double hypothesis[d];

	for (i = 0; i < d; i++)
		hypothesis[i] = 0;

	for (i = 0; i < m; i++) {
		if (*(hostX + i) < 0) {
			lambda += *(hostX + i);
			for (j = 0; j < d; j++)
				hypothesis[j] += *(hostX + i) * data[i + 1][j];
		}
		else
			continue;
	}*/
