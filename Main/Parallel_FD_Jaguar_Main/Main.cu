#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#define SizeX 1024
#define SizeY 1024

/* run this program using the console pauser or add your own getch, system("pause") or input loop */

// Non-CUDA stuffs
void show(double* x);
void writeText(double *u);
void readmatrix(double* a, const char* filename);

// CUDA stuffs
//  get k value
__global__
void getK(double *u1, double *v1, double *ku, double *kv) {
	// Working area
	int Nx = blockIdx.x * blockDim.x + threadIdx.x;
	int Ny = blockIdx.y * blockDim.y + threadIdx.y;
	int i = Nx + Ny * blockDim.x * gridDim.x;

	double dx;
	dx = 1.0;

	// Initialize hyperparameter
	double a = 0.45 * 6, b = 6, alp = 0.899, bet = -0.91, gam = -alp, r2 = 2.0, r3 = 3.5;

	double Ztop, Zleft, Zbottom, Zright, Zcenter;
	double deltau, deltav;

	//laplacian
	if (i > SizeY && i < SizeX * SizeY - SizeY && i % SizeY != 0 && i % SizeY != SizeY - 1) {
		Ztop = u1[i - SizeY];
		Zleft = u1[i - 1];
		Zbottom = u1[i + SizeY];
		Zright = u1[i + 1];
		Zcenter = u1[i];
		deltau = (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / pow(dx, 2);
		//printf("\n deltau ke-%d = %f", i, deltau);
		//printf("\n u ke-%d = %f", i, dx);

		Ztop = v1[i - SizeY];
		Zleft = v1[i - 1];
		Zbottom = v1[i + SizeY];
		Zright = v1[i + 1];
		Zcenter = v1[i];
		deltav = (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / pow(dx, 2);
	} else {
		deltau = u1[i];
		deltav = v1[i];
	}

	// Update
	ku[i] = (a * deltau + alp * u1[i] + v1[i] - r2 * u1[i] * v1[i]
		- alp * r3 * u1[i] * v1[i] * v1[i]);
	kv[i] = (b * deltav + gam * u1[i] + bet * v1[i] + r2 * u1[i] * v1[i]
		+ alp * r3 * u1[i] * v1[i] * v1[i]);
	//printf("\n ku ke-%d = %f", i, ku[i]);
}

__global__
void Neumann(double *u1, double *v1) {
	int Nx = blockIdx.x * blockDim.x + threadIdx.x;
	int Ny = blockIdx.y * blockDim.y + threadIdx.y;
	int i = Nx + Ny * blockDim.x * gridDim.x;

	if (i < SizeY) {
		u1[i] = u1[i + SizeY];
		v1[i] = v1[i + SizeY];
	}
	else if (i > SizeX*SizeY - SizeY) {
		u1[i] = u1[i - SizeY];
		v1[i] = v1[i - SizeY];
	}
	else if (i % SizeY == 0) {
		u1[i] = u1[i + 1];
		v1[i] = v1[i + 1];
	}
	else if (i % SizeY == SizeY - 1) {
		u1[i] = u1[i - 1];
		v1[i] = v1[i - 1];
	}
}

// Update U and V value
__global__
void update(double dt, double *k1u, double *k1v, double *u, double *v, double *u1, double *v1, double *ku, double *kv) {
	int Nx = blockIdx.x * blockDim.x + threadIdx.x;
	int Ny = blockIdx.y * blockDim.y + threadIdx.y;
	int i = Nx + Ny * blockDim.x * gridDim.x;
	
	u1[i] = u[i] + dt * (ku[i] + k1u[i]);
	v1[i] = v[i] + dt * (kv[i] + k1v[i]);
}

// Main Program
int main(int argc, char *argv[]) {
	// Index
	int i, j, k;
	double dx, dy;
	dx = (double) 2 / (SizeX - 1);
	dy = (double) 2 / (SizeY - 1);

	// 	Matrix size of x, y, and u with Nx as size of x and Ny as size of y
	int Nx = SizeX, Ny = SizeY;
	int elem = Nx*Ny;
	
	dim3 block(32 / 4, 32 / 4);
	dim3 grid((SizeY + block.x - 1) / block.x, (SizeX + block.y - 1) / block.y);

	// Time
	//double dt = 0.000005;
	double dt = 0.01;

	// Intialize array matrix of x, y, u, v, k, k1, and Area
	double *u, *v, *u1, *v1, *ku, *kv, *k1u, *k1v;
	double *d_u, *d_v, *d_u1, *d_v1, * d_ku, * d_kv, * d_k1u, * d_k1v;

	u = (double*)malloc(elem*sizeof(double));
	v = (double*)malloc(elem*sizeof(double));
	u1 = (double*)malloc(elem*sizeof(double));
	v1 = (double*)malloc(elem*sizeof(double));
	ku = (double*)malloc(elem * sizeof(double));
	kv = (double*)malloc(elem * sizeof(double));
	k1u = (double*)malloc(elem * sizeof(double));
	k1v = (double*)malloc(elem * sizeof(double));
	
	cudaMalloc(&d_u, elem*sizeof(double));
	cudaMalloc(&d_v, elem*sizeof(double));
	cudaMalloc(&d_u1, elem*sizeof(double));
	cudaMalloc(&d_v1, elem*sizeof(double));
	cudaMalloc(&d_ku, elem * sizeof(double));
	cudaMalloc(&d_kv, elem * sizeof(double));
	cudaMalloc(&d_k1u, elem * sizeof(double));
	cudaMalloc(&d_k1v, elem * sizeof(double));

	for (i = 0;i < elem;i++) {
		u[i] = rand() % 2;
		v[i] = rand() % 2;
	}

	//readmatrix(u, "u.txt");
	//readmatrix(v, "v.txt");

	// Send initialized values to Device
	cudaMemcpy(d_u, u, elem * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v, elem * sizeof(double), cudaMemcpyHostToDevice);

	// Another magic batch!
	cudaMemset(d_ku, 0, elem * sizeof(double));
	cudaMemset(d_kv, 0, elem * sizeof(double));

	cudaMemcpy(d_u1, d_u, elem * sizeof(double), cudaMemcpyHostToHost);
	cudaMemcpy(d_v1, d_v, elem * sizeof(double), cudaMemcpyHostToHost);

	// Vibe Check
	//cudaMemcpy(u1, d_u1, elem * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(v1, d_v1, elem * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(ku, d_ku, elem * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(kv, d_kv, elem * sizeof(double), cudaMemcpyDeviceToHost);

	//cudaMemcpy(u, d_u, elem * sizeof(double), cudaMemcpyDeviceToHost);
	//printf("\n\n\n U ");
	//show(u);

	//cudaMemcpy(v, d_v, elem * sizeof(double), cudaMemcpyDeviceToHost);
	//printf("\n\n\n V ");
	//show(v);

	//cudaMemcpy(x, d_x, elem * sizeof(double), cudaMemcpyDeviceToHost);
	//printf("\n\n\n x ");
	//show(x);

	//cudaMemcpy(y, d_y, elem * sizeof(double), cudaMemcpyDeviceToHost);
	//printf("\n\n\n y ");
	//show(y);

	// Record Time
	float elapsedTime = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaSetDevice(0);
	cudaEventRecord(start, 0);

	//double save[30000 / 3000][SizeX * SizeY];
	//i = 0;

	for (k = 0; k < 30000; k++) {
		printf("\n %d", k);
		// zero-ing temp
		cudaMemset(d_k1u, 0, elem * sizeof(double));
		cudaMemset(d_k1v, 0, elem * sizeof(double));

		// Copy U and V to U1 and V1
		cudaMemcpy(d_u1, d_u, elem * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_v1, d_v, elem * sizeof(double), cudaMemcpyDeviceToDevice);



		// =============================================================================
		// K1 Stuffs
		// get K1

		getK << <grid, block >> > (d_u1, d_v1,d_ku, d_kv);

		// Update U1 and V1 first
		update << <grid, block >> > (dt / 1.0, d_k1u, d_k1v, d_u, d_v, d_u1, d_v1, d_ku, d_kv);

		// Neumann
		Neumann << <grid, block >> > (d_u1, d_v1);

		// Vibe Check
		//cudaMemcpy(u1, d_u1, elem * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(v1, d_v1, elem * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(ku, d_ku, elem * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(kv, d_kv, elem * sizeof(double), cudaMemcpyDeviceToHost);

		//printf("\n U1 - %d", k);
		//show(u1);
		//printf("\n V1 - %d", k);
		//show(v1);
		//printf("\n K1U - %d", k);
		//show(ku);
		//printf("\n K1V - %d", k);
		//show(kv);
		// =============================================================================



		// =============================================================================
		// K2 Stuffs
		// allow move  ku value to k1u so ku can be come k2u (and so with kv)
		cudaMemcpy(d_k1u, d_ku, elem * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_k1v, d_kv, elem * sizeof(double), cudaMemcpyDeviceToDevice);

		// get K2
		//cudaMemcpy(d_Area, Area, elem * sizeof(double), cudaMemcpyHostToDevice);

		getK << <grid, block >> > (d_u1, d_v1, d_ku, d_kv);

		//cudaMemcpy(ku, d_ku, elem * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(kv, d_kv, elem * sizeof(double), cudaMemcpyDeviceToHost);

		// Update final U and V as U1 and V1

		update << <grid, block >> > (dt / 2.0, d_k1u, d_k1v, d_u, d_v, d_u1, d_v1, d_ku, d_kv);


		// Neumann
		Neumann << <grid, block >> > (d_u1, d_v1);

		// Vibe Check
		//cudaMemcpy(u1, d_u1, elem * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(v1, d_v1, elem * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(ku, d_ku, elem * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(kv, d_kv, elem * sizeof(double), cudaMemcpyDeviceToHost);

		//printf("\n U2 - %d", k);
		//show(u1);
		//printf("\n V2 - %d", k);
		//show(v1);
		//printf("\n K2U - %d", k);
		//show(ku);
		//printf("\n K2V - %d", k);
		//show(kv);
		// =============================================================================



		// Finishing
		// get U and V final result from U1 and V1
		cudaMemcpy(d_u, d_u1, elem * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_v, d_v1, elem * sizeof(double), cudaMemcpyDeviceToDevice);
		/*if (k % 3000 == 0) {
			cudaMemcpy(u, d_u, elem * sizeof(double), cudaMemcpyDeviceToHost);
			for (j = 0;j < elem;j++) {
				save[i][j] = u[j];
			}
			i++;
		}*/
	}

	// CUDA synchronize and get time
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);

	// Vibe Check
	cudaMemcpy(u, d_u, elem * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(v, d_v, elem * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(u1, d_u1, elem * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(v1, d_v1, elem * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(ku, d_ku, elem * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(kv, d_kv, elem * sizeof(double), cudaMemcpyDeviceToHost);

	printf("\n\n\n U - Update");
	show(u);

	printf("\n\n\n V - Update");
	show(v);

	printf("\n\n\n Lama Proses GPU = %.4fms", elapsedTime);

	writeText(u);
	cudaFree(d_u); cudaFree(d_v);
	cudaFree(d_u1); cudaFree(d_v1);
	cudaFree(d_ku); cudaFree(d_kv);
	free(u); free(v);
	free(u1); free(v1);
	free(ku); free(kv);

	return 0;
}


// Non-CUDA stuffs
void show(double* x) {
	for (int i = 0; i < SizeX * SizeY; i++) {
		if (i % SizeY == 0) {
			printf("\n[%.4f", x[i]);
		}
		else if (i % SizeY == SizeY - 1) {
			printf(", %.4f];", x[i]);
		}
		else {
			printf(", %.4f", x[i]);
		}
	}
}

// create .txt matrix result
void writeText(double *u){
	int i, j;
	
	const char *filename = "Matrix_Result.txt";

    // open the file for writing
    FILE *fp = fopen(filename, "w");
    // write to the text file
    for (i = 0; i < SizeX*SizeY; i++){
    	if(i%SizeY == 0){
		   	fprintf(fp, "\n[%.4f",u[i]);
		}else if(i%SizeY == SizeY-1){
			fprintf(fp, ", %.4f];",u[i]);
		}else{
			fprintf(fp, ", %.4f",u[i]);
		}
	}

    // close the file
    fclose(fp);

    return;
}

void readmatrix(double* a, const char* filename)
{
	int i, j;

	FILE* pf;
	pf = fopen(filename, "r");
	if (pf == NULL)
		return;

	for (i = 0; i < SizeX * SizeY; i++)
	{
		for (j = 0; j < SizeY; ++j) {
			fscanf(pf, "%lf", &a[i * SizeY + j]);
		}
	}

	fclose(pf);
	return;
}