//#include "../common/book.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>

/*This example finds a cuda device with compute capabilty 1.3 or higher*/

int main(void) 
{
	//Set a cuda device prop
	cudaDeviceProp prop;
	int dev;

	//Get id of current CUDA device
	cudaGetDevice(&dev);
	printf("ID of current CUDA device: %d\n", dev);

	//Set cuda device properties to what we need
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 3;

	//Choose a device closest to prop
	cudaChooseDevice(&dev, &prop);
	printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
	cudaSetDevice(dev);
}