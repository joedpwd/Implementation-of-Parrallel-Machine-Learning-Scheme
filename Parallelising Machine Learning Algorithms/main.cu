#include "main.h"

int main(int argc, char **argv) {

	//Used for Iteration
	int i;
	int j;
	int k;

	//Get the dimensions of the data
	//Set Valuues
	//h will be a hyper parameter passed to the program.

	const int d = 2;

	const int r = d + 2; //Assuming d = 2
	
	const int h = 1; //Hyper parameter

	const int m = d + 1; //Equivalent to d + 1

	const int rh = pow(r, h);

	//Perform rh instances of ML problem

	double luSolverA[100 * m];
	double luSolverB[m];

	const int lda = m;
	const int ldb = m;

	double *data = (double *)malloc(sizeof(double) * rh * d);

	i = 0;

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

	/*(data) = 141.83527105;
	*(data+1) = 813.5546238;
	*(data+2) = 156.99282842;
	*(data+3) = 952.02261359;
	*(data+4) = 155.03418909;
	*(data+5) = 971.72707688;
	*(data+6) = 153.36539951;
	*(data+7) = 974.17111871;*/

	/*(data) = 139.82134402;
	*(data+1) = 462.9137674;
	*(data+2) = 154.98607891;
	*(data+3) = 1334.30439163;
	*(data+4) = 134.32358629;
	*(data+5) = 329.4845056;
	*(data+6) = 127.43249724;
	*(data+7) = 784.10491373;
	*(data+8) = 183.22158548;
	*(data+9) = 892.03897951;
	*(data+10) = 138.60017152;
	*(data+11) = 970.10978656;
	*(data+12) = 161.45365155;
	*(data+13) = 1160.31093851;
	*(data+14) = 155.50481064;
	*(data+15) = 805.66838716;
	*(data+16) = 155.88761615;
	*(data+17) = 1299.67696078;
	*(data+18) = 145.90620091;
	*(data+19) = 1167.34228914;
	*(data+20) = 155.88139939;
	*(data+21) = 882.92696539;
	*(data+22) = 167.63707993;
	*(data+23) = 395.28292253;
	*(data+24) = 167.12573196;
	*(data+25) = 1418.2905522;
	*(data+26) = 152.20922953;
	*(data+27) = 679.764223;
	*(data+28) = 163.72618895;
	*(data+29) = 969.27309617;
	*(data+30) = 134.50651337;
	*(data+31) = 994.33789174;*/

	for (i = 0; i < rh * d; i++) {
		printf("%.10f\n", *(data + i));
	}

	for (i = 0; i < h; i++) {
		for (j = 0; j < pow(r, h-1 - i); j++) {
			radonInstance((data + (d*j*r)), d);
		}
		for (j = 0; j < pow(r, h-1 - i); j++) {
			for(k=0;k<d;k++)
				*(data + (j*d) + k) = *(data + (r*j*d) + k);
		}
	}

	/*for (i = 0; i < d; i++) {
		printf("%.5f\n", *(data + i));
	}*/

	/*for (i = 0; i < rh; i++) {
		data[i][0] = rand() % 50;
		data[i][1] = rand() % 20;
		data[i][2] = i * i + 1;
		//printf("%.2lf %.2lf %.2lf\n", data[i][0], data[i][1], data[i][2]);
	}*/
	//Until we get data just plug in the same data
	/*for (j = 0; j < d; j++) {
		luSolverB[j] = -*(data + j);
	}

	luSolverB[j] = -1;

	for (i = 1; i < rh; i++) {
		for (j = 0; j < d; j++) {
			luSolverA[(i-1)*m + j] = *(data + (i*d) + j);
		}
		luSolverA[(i - 1)*m + j] = 1;
	}

	double *hostA = (double *)(malloc(m*m * sizeof(double)));
	double *hostB = (double *)(malloc(m * sizeof(double)));
	double *hostX = (double *)(malloc(m * sizeof(double)));
	double *hostLU = (double *)(malloc(m*m * sizeof(double)));
	int *hostIpiv = (int *)(m * malloc(sizeof(int)));
	int *hostInfo = (int *)(malloc(sizeof(int)));

	for (i = 0; i < m*m; i++) {
		*(hostA + i) = luSolverA[i];
		printf("%.2lf\n", luSolverA[i]);
	}
	for (i = 0; i < m; i++)
		*(hostB + i) = luSolverB[i];


	denseLUSolver(hostA, hostB, hostX, hostLU, hostIpiv, hostInfo, m);*/

	/*for (i = 0; i < m; i++)
		printf("X\t%d\t%.4f\n", i, *(hostX+i));*/

	/*double lambda = 1;
	double hypothesis[d];
	
	for (i = 0; i < d; i++)
		hypothesis[i] = *(data + i);

	for (i = 0; i < m; i++) {
		if (*(hostX + i) >= 0) {
			lambda += *(hostX + i);
			for (j = 0; j < d; j++)
				hypothesis[j] += *(hostX + i) * *(data + ((i+1)*d) + j);
		}
		else
			continue;
	}

	printf("RESULT\n");*/

	/*for (i = 0; i < d; i++) {
		printf("%.5f\n", hypothesis[i]);
	}*/

	/*for (i = 0; i < d; i++) {
		hypothesis[i] /= lambda;
		//printf("%.5f\n", lambda);
		printf("%.5f\n", hypothesis[i]);
	}*/
	free(test);
	free(data);
}

void radonInstance(double *dataPoints, const int d) {

	int i, j;

	const int r = d + 2; //Assuming d = 2

	const int m = d + 1;

	double *luSolverA = (double *)malloc(sizeof(double)*m*m);
	double *luSolverB = (double *)malloc(sizeof(double)*m);

	for (j = 0; j < d; j++) {
		luSolverB[j] = -*(dataPoints+j);
	}

	luSolverB[j] = -1;

	for (i = 1; i < r; i++) {
		for (j = 0; j < d; j++) {
			luSolverA[(i - 1)*m + j] = *(dataPoints + (i*d) + j);
		}
		luSolverA[(i - 1)*m + j] = 1;
	}


	double *hostA = (double *)(malloc(m*m * sizeof(double)));
	double *hostB = (double *)(malloc(m * sizeof(double)));
	double *hostX = (double *)(malloc(m * sizeof(double)));
	double *hostLU = (double *)(malloc(m*m * sizeof(double)));
	int *hostIpiv = (int *)(malloc(m * sizeof(int)));
	int *hostInfo = (int *)(malloc(sizeof(int)));

	long long *Acopy;
	long long *Bcopy;
	long long *Xcopy;

	for (i = 0; i < m*m; i++) {
		*(hostA + i) = luSolverA[i];
		//printf("%.2lf\n", luSolverA[i]);
	}
	for (i = 0; i < m; i++)
		*(hostB + i) = luSolverB[i];


	denseLUSolver(hostA, hostB, hostX, hostLU, hostIpiv, hostInfo, m);

	/*for (i = 0; i < m; i++)
		printf("X\t%d\t%.4f\n", i, *(hostX+i));*/

	double lambda = 1;
	double *hypothesis = (double *)malloc(sizeof(double)*d);

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

	std::ofstream A, B, X;
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
	}

	//free(Acopy);
	//free(Bcopy);
	//free(Xcopy);

	free(luSolverA);
	free(luSolverB);
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
