#include <stdio.h>
#include "../common/book.h"

__global__ void kernal( void )
{
}

int main(void) 
{
	kernal<<<1,1>>>();
	printf("Hello, World Kernal Call\n");
	return 0;
}