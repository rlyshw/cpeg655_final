#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 8
#define logn 3
#define threadsPerBlock 9

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

__global__ void dft(double*x, double*Xre, double*Xim){
	//Credit to Shengfeng Chen from 
	//https://cs.wmich.edu/gupta/teaching/cs5260/5260Sp15web/studentProjects/IMPLEMENTATION%20of%20DFT%20in%20CPU%20and%20GPU%20by%20Shengfeng.pdf
	__shared__ double cache[2*(N-1)];
	int n = threadIdx.x, k=blockIdx.x, cacheIndex = threadIdx.x;
//	Matrix computation for Xim and Xre
	double temp1=0,temp2=0;
	while(n<N && k<N){
		temp1 += x[n] * cos(n*k*(M_PI*2) / N);
		temp2 -= x[n] * sin(n*k*(M_PI*2) / N);
		n+=N; k+=N;
	}

	cache[cacheIndex] = temp1;
	cache[cacheIndex+blockDim.x] = temp2;
	__syncthreads();
	
	int i =blockDim.x/2;
	while(i!=0){
	if(cacheIndex<i){
		cache[cacheIndex]+=cache[cacheIndex+i];
		cache[blockDim.x+cacheIndex]+=cache[blockDim.x+cacheIndex+i];}
	__syncthreads();
	i/=2;
	}
	if(cacheIndex == 0){
		Xre[blockIdx.x] = cache[0];
		Xim[blockIdx.x] = cache[blockDim.x];}
}
__global__ void fft(double*x, double*Xre, double*Xim){
	int tx = threadIdx.x, ty=threadIdx.y;
	double temp = 0;
	const int cache_size = logn*N;
	__shared__ double cache[cache_size];

	if(tx==(logn-1)){
		int ty_rev = __brev(ty)>>(32-logn);
		if ((ty%2)==0) temp = x[ty_rev]+x[ty_rev+N/2];
		else if ((ty%2)!=0) temp = x[ty_rev-N/2]-x[ty_rev];
		//printf("x[%d]:%f tx,ty:%d,%d val:%f ty_rev:%d\n",ty,x[ty],tx,ty,temp,ty_rev);
	}
	else{	
		int n_prime = (int)N/powf(2,tx);
		int offset;
		if(ty<(n_prime/2))
			offset = ty+(n_prime/2);
		else
			offset = ty%(n_prime/2);
		float TF;
		TF = cos(tx*ty*(M_PI*2) / N);
		//printf("tx,ty:%d,%d n_prime:%d offset:%d\n",tx,ty,n_prime,offset);
		__syncthreads();
		if((ty%2)==0) temp = cache[tx+1+logn*ty]+cache[tx+1+logn*offset]*TF;
		if((ty%2)!=0) temp = cache[tx+1+logn*ty]-cache[tx+1+logn*offset]*TF;
		//printf("tx,ty:%d,%d val:%f\n",tx,ty,temp);
	}

	cache[tx+logn*ty] = temp;
	Xre[ty] = cache[logn*ty];
	__syncthreads();
}

int main(){

	int i;
	
	double *d_X, *d_Xre, *d_Xim;	
	double *h_X, *h_Xre, *h_Xim;

	size_t size = N*sizeof(double);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Allocate Device memory
	cudaMalloc((void **)&d_X, size);
	cudaMalloc((void **)&d_Xre, size);
	cudaMalloc((void **)&d_Xim, size);

	//Allocate Host memory
	cudaMallocHost((void **)&h_X, size);
	cudaMallocHost((void **)&h_Xre, size);	
	cudaMallocHost((void **)&h_Xim, size);

	
	//Initialize matrices on the host
	for(i=0;i<N;i++){
		h_X[i]=i;//sin(i);
	}


	//Allocate X to the Device
	cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);


	//Invoke kernel
	dim3 blockPerGrid(1);
	dim3 threadPerBlock(logn,N);

	//cudaEventRecord(start);	
	//MatrixMultiply<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, d_C);
	//cudaEventRecord(stop);	

	cudaEventRecord(start);	
	fft<<<blockPerGrid, threadPerBlock>>>(d_X, d_Xre, d_Xim);
	cudaEventRecord(stop);	


	//Read from device
	cudaMemcpy(h_Xre, d_Xre, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Xim, d_Xim, size, cudaMemcpyDeviceToHost);
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("%f\n", milliseconds);
	
	//Calculate the MM result with normal CPU implementation and compare the results with the GPU

	float * test_Xre;
	float * test_Xim;
	test_Xre = (float *)malloc(size);	
	test_Xim = (float *)malloc(size);	
	for (int k=0;k<N;k++){
		test_Xre[k]=0;
		test_Xim[k]=0;
		for(int n=0;n<N;n++){
			test_Xre[k]+=h_X[n]*cos(n*k*M_PI*2 / N);
			test_Xim[k]-=h_X[n]*sin(n*k*M_PI*2 / N);
		}
	}
	int compare_Xre = 0;
	int compare_Xim = 0;
	for(i=0;i<N;i++){
		printf("XRE: test: %f comp: %f\n",test_Xre[i],h_Xre[i]);
		//printf("XIM: test: %f comp: %f\n\n",test_Xim[i],h_Xim[i]);
		if(test_Xre[i]==h_Xre[i]) compare_Xre++;
		if(test_Xim[i]==h_Xim[i]) compare_Xim++;
	}
	if(compare_Xre == N){// && compare_Xim==N){
		printf("Success!\n");
	}else{
		printf("Error!\n");	
	}

	/*=============================Finish Test=================================*/

	free(test_Xre);
	free(test_Xim);
	cudaFree(d_X);
	cudaFree(d_Xre);
	cudaFree(d_Xim);
	cudaFree(h_X);
	cudaFree(h_Xre);
	cudaFree(h_Xim);
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
