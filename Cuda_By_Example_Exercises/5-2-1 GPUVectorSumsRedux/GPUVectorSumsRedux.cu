//#include "../common/book.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>

#define N (33 * 1024)

__global__ void add(int *a, int *b, int *c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x; // handle the data at this index
	while (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int main(void) 
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	//allocate the memory on the GPU
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	//Fill the arrays a and b on the CPU
	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	//Copy the arrays a and b to the GPU
	cudaMemcpy(dev_a, a, N *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N *sizeof(int), cudaMemcpyHostToDevice);

	add<<<128,128>>>(dev_a, dev_b, dev_c);

	//Copy c back from the GPU to CPU
	cudaMemcpy(c, dev_c, N *sizeof(int), cudaMemcpyDeviceToHost);

	//Display results
	for (int i = 0; i < N; i++)
		printf("%d + %d = %d\n", a[i], b[i], c[i]);

	//Verify GPU was succesful ro failed
	bool success = true;
	for (int i =0; i < N; i++)
	{
		if((a[i] + b[i] != c[i]))
		{
			printf("Error: %d + %d != %d\n", a[i], b[i], c[i]);
			success = false;
		}
	}
	if(success) printf("We did it!!\n");

	//Free the memory allocated to the GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}