#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define SizeX 1024
#define SizeY 1024

/* run this program using the console pauser or add your own getch, system("pause") or input loop */

void writeText(double u[SizeX*SizeY]);
void readmatrix(double *a, const char* filename);

int main(int argc, char *argv[]) {
	clock_t start, end;
	// Index
	long int i, j, k;
	
	// 	Matrix size of x, y, and u with Nx as size of x and Ny as size of y
	int Nx = SizeX, Ny = SizeY;
	long int elem = Nx*Ny;
	
	double dx, dy;
	dx = 1.0;
	dy = 1.0;
	
	// Initialize hyperparameter
	double a = 0.45*6, b = 6, alp = 0.899, bet = -0.91, gam = -alp, r2 = 2.0, r3 = 3.5;
	
	// Intialize array matrix of x, y, u, and v
	double *u, *v, *u1, *v1;

	u = (double*)malloc(elem * sizeof(double));
	v = (double*)malloc(elem * sizeof(double));
	u1 = (double*)malloc(elem * sizeof(double));
	v1 = (double*)malloc(elem * sizeof(double));
	
	for (i = 0;i < elem;i++) {
		u[i] = rand() % 2;
		v[i] = rand() % 2;
	}
	printf("\n 3");
	
//	readmatrix(u,"u.txt");
//	readmatrix(v,"v.txt");
	
	for (i = 0; i < elem; i++) {
		u1[i] = u[i];
		v1[i] = v[i];
//		printf("%.4f",u[i]);	
	}
	
	int time = 0;
	double dt = 0.01;
	
	// Length
	int elem_c = (Nx-1)*(Ny-1);
	int elemc = (Nx-2)*(Ny-2);
	
	// Working area
	double Ztop, Zleft, Zbottom, Zright, Zcenter;
	double deltau, deltav;
	double *k1u, *k1v, *k2u, *k2v;
	

	k1u = (double*)malloc(elemc * sizeof(double));
	k1v = (double*)malloc(elemc * sizeof(double));
	k2u = (double*)malloc(elemc * sizeof(double));
	k2v = (double*)malloc(elemc * sizeof(double));
	
	int pad;
	
	start = clock();
	
	for (k = 0; k < 30000; k++){
		printf("\n %d",k);
		pad = 0;
		for (i = 0; i < elemc; i++){
			if (i%(Ny-2) == 0 && i != 0){
				pad += 2;
			}
			Ztop = u[i+1+pad];
			Zleft = u[i+Ny+pad];
			Zbottom = u[i+2*Ny+1+pad];
			Zright = u[i+Ny+2+pad];
			Zcenter = u[i+Ny+1+pad];
			deltau = (Ztop + Zleft + Zbottom + Zright - 4*Zcenter)/pow(dx,2.0);
//			printf(" %.4f,",deltau);
			
			Ztop = v[i+1+pad];
			Zleft = v[i+Ny+pad];
			Zbottom = v[i+2*Ny+1+pad];
			Zright = v[i+Ny+2+pad];
			Zcenter = v[i+Ny+1+pad];
			deltav = (Ztop + Zleft + Zbottom + Zright - 4*Zcenter)/pow(dx,2.0);
			
			// Update
			k1u[i] = (a*(deltau) + alp*u[i+Ny+1+pad] + v[i+Ny+1+pad] - r2*u[i+Ny+1+pad]*v[i+Ny+1+pad] 
							 - alp*r3*u[i+Ny+1+pad]*v[i+Ny+1+pad]*v[i+Ny+1+pad]);
//			printf(" %.4f,",u1[i+Ny+1+pad]);

			k1v[i] = (b*(deltav) + gam*u[i+Ny+1+pad] + bet*v[i+Ny+1+pad] + r2*u[i+Ny+1+pad]*v[i+Ny+1+pad] 
							 + alp*r3*u[i+Ny+1+pad]*v[i+Ny+1+pad]*v[i+Ny+1+pad]);
							 
			u1[i+Ny+1+pad] = u[i+Ny+1+pad] + dt*k1u[i];
			v1[i+Ny+1+pad] = v[i+Ny+1+pad] + dt*k1v[i];
		}
		
		// Neumann
		for (i = 0; i < elem; i++){
			if (i < Ny){
				u1[i] = u1[i+Ny];
				v1[i] = v1[i+Ny];
			} else if (i > elem-Ny){
				u1[i] = u1[i-Ny];
				v1[i] = v1[i-Ny];
			} else if (i%Ny == 0){
				u1[i] = u1[i+1];
				v1[i] = v1[i+1];
			} else if (i%Ny == Ny-1){
				u1[i] = u1[i-1];
				v1[i] = v1[i-1];
			}
		}
		
		pad=0;
		for (i = 0; i < elemc; i++){
			if (i%(Ny-2) == 0 && i != 0){
				pad += 2;
			}
			Ztop = u1[i+1+pad];
			Zleft = u1[i+Ny+pad];
			Zbottom = u1[i+2*Ny+1+pad];
			Zright = u1[i+Ny+2+pad];
			Zcenter = u1[i+Ny+1+pad];
			deltau = (Ztop + Zleft + Zbottom + Zright - 4*Zcenter)/pow(dx,2.0);
			
			Ztop = v1[i+1+pad];
			Zleft = v1[i+Ny+pad];
			Zbottom = v1[i+2*Ny+1+pad];
			Zright = v1[i+Ny+2+pad];
			Zcenter = v1[i+Ny+1+pad];
			deltav = (Ztop + Zleft + Zbottom + Zright - 4*Zcenter)/pow(dx,2.0);
			
			// Update
			k2u[i] = (a*(deltau) + alp*u1[i+Ny+1+pad] + v1[i+Ny+1+pad] - r2*u1[i+Ny+1+pad]*v1[i+Ny+1+pad] 
							 - alp*r3*u1[i+Ny+1+pad]*v1[i+Ny+1+pad]*v1[i+Ny+1+pad]);
			
			k2v[i] = (b*(deltav) + gam*u1[i+Ny+1+pad] + bet*v1[i+Ny+1+pad] + r2*u1[i+Ny+1+pad]*v1[i+Ny+1+pad] 
							 + alp*r3*u1[i+Ny+1+pad]*v1[i+Ny+1+pad]*v1[i+Ny+1+pad]);
		}
		
		pad=0;
		for (i = 0; i < elemc; i++){
			if (i%(Ny-2) == 0 && i != 0){
				pad += 2;
			}
			u1[i+Ny+1+pad] = u[i+Ny+1+pad] + (dt/2.0)*(k1u[i] + k2u[i]);
			v1[i+Ny+1+pad] = v[i+Ny+1+pad] + (dt/2.0)*(k1v[i] + k2v[i]);
		}
		
		// Neumann
		for (i = 0; i < elem; i++){
			if (i < Ny){
				u1[i] = u1[i+Ny];
				v1[i] = v1[i+Ny];
			} else if (i > elem-Ny){
				u1[i] = u1[i-Ny];
				v1[i] = v1[i-Ny];
			} else if (i%Ny == 0){
				u1[i] = u1[i+1];
				v1[i] = v1[i+1];
			} else if (i%Ny == Ny-1){
				u1[i] = u1[i-1];
				v1[i] = v1[i-1];
			}
		}
		
		//u = u1; v = v1;
		for (i = 0; i < elem; i++) {
	    	u[i] = u1[i];
	    	v[i] = v1[i];
	    }
	}
	
	end = clock();
	
	for (i = 0; i < elem; i++) {
	    if(i%Ny == 0){
	    	printf("\n[%.4f",u1[i]);
		}else if(i%Ny == Ny-1){
			printf(", %.4f];",u1[i]);
		}else{
			printf(", %.4f",u1[i]);
		}
	}
	
	double time_spent;
	time_spent = (double) end - start;
	printf("\n\n\n Lama Proses CPU = %.4fms", time_spent);
	writeText(u);
	//printf("1234");
	
	return 0;
}

void writeText(double u[SizeX*SizeY]){
	int i, j;
	
	char *filename = "Matrix_Result.txt";

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

void readmatrix(double *a, const char* filename)
{
	int i,j;

	FILE* pf;
	pf = fopen(filename, "r");
	if (pf == NULL)
		return;

	for (i = 0; i < SizeX*SizeY; i++)
	{
		for (j = 0; j < SizeY; ++j){
			fscanf(pf, "%lf", &a[i*SizeY + j]);
		}
	}

	fclose(pf);
	return;
}
