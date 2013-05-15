#include "../common/book.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//Declare method with __global to indicate
//function should be compiled to run on a device
__global__ void add(int a, int b, int *c) 
{
	//Add 2 numbers together and store in location pointed by *c
	*c = a + b;
}

int main(void)
{
	//Declare variables for holding data
	int c; int a = 100; int b = 50;
	int *dev_c;

	/*
	 Allocate memory on the device(GPU)
	 param one is a pointer to a pointer you want to hold the address
	 and param 2 is the size of the memory allocation.

	 Hanlde Error is a utility macro to detect any errors and exit application.
	*/
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

	//call the add method, passing parameters
	add<<<1,1>>>(a, b, dev_c);

	//Copy the memory from the device to the host so the data can be used by the host
	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

	//Print the result
	printf("%d + %d = %d\n", a, b, c);

	//Free the memory
	cudaFree(dev_c);

	return 0;
}