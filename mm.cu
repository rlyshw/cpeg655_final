#include <stdio.h>
#include <stdlib.h>

#define N 22 

__global__ void MatrixMultiply(float *d_A, float *d_B, float *d_C)
{
	
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	
	float Cvalue = 0.0;	
	int k;


	for(k=0; k<N; k++){
		float d_Aelment = d_A[ty*N + k];
		float d_Belment = d_B[k*N + tx];
		Cvalue += d_Aelment * d_Belment;		
	}
	d_C[ty*N + tx]= Cvalue;
}

int main(){

	int i,j,k;
	
	float *d_A, *d_B, *d_C;	
	float *h_A, *h_B, *h_C;

	size_t size = N*N*sizeof(float);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Allocate Device memory
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);

	//Allocate Host memory
	cudaMallocHost((void **)&h_A, size);
	cudaMallocHost((void **)&h_B, size);	
	cudaMallocHost((void **)&h_C, size);

	
	//Initialize matrices on the host
	for(i=0;i<N;i++){
	    for(j=0;j<N;j++){
		h_A[i*N+j]=i;
		h_B[i*N+j]=i+1;
	    }
	}


	//Allocate A and B to the Device
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


	//Invoke kernel
	dim3 blockPerGrid(1,1);
	dim3 threadPerBlock(N,N);

	cudaEventRecord(start);	
	MatrixMultiply<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, d_C);
	cudaEventRecord(stop);	


	//Read C from device
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("%f\n", milliseconds);
	
	//Calculate the MM result with normal CPU implementation and compare the results with the GPU

	float * test_C;
	test_C = (float *)malloc(size);	
	for (i=0; i<N; i++){
		for (j=0;j<N;j++){
			float sum = 0;
			for (k=0;k<N;k++){
				float a = h_A[i*N+k];
				float b = h_B[k*N+j];
				sum += a*b;
			}
			test_C[i*N+j]= sum;
		}
	}
	int compare = 0;
	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			if (test_C[i*N+j]==h_C[i*N+j]){
				compare++;
			}else{
				compare+=0;
			}
		}
	}
	if(compare == N*N){
		//printf("Success!\n");
	}else{
		//printf("Error!\n");	
	}

	/*=============================Finish Test=================================*/

	free(test_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(h_A);
	cudaFree(h_B);
	cudaFree(h_C);
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
