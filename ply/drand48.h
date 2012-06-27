#ifndef DRAND48_H  
#define DRAND48_H  

#include <stdlib.h>  

#define m 0x100000000LL  
#define _c 0xB16  
#define a 0x5DEECE66DLL  

static unsigned long long seed = 1;  

double drand48(void)  
{  
	unsigned int x = seed >> 16;  

	seed = (a * seed + _c) & 0xFFFFFFFFFFFFLL;  
	return  ((double)x / (double)m);  

}  

void srand48(unsigned int i)  
{  
	seed  = (((long long int)i) << 16) | rand();  
} 

#endif