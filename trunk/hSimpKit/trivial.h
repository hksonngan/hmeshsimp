/*
 *  some trivial operations
 *
 *  author : ht
 *  email  : waytofall916@gmail.com
 */

#ifndef __TRIVIAL__
#define __TRIVIAL__

#include <string>

using std::string;

// specify the endian order
enum EndianOrder { H_BIG_ENDIAN, H_LITTLE_ENDIAN };

// trim the extention of the file name and append some str
void trimExtAndAppend(char *src, char *dst, char *app);

// get the system endian mode
EndianOrder getSystemEndianMode();

// switch bytes for a variable
inline void switchBytes(char* ptr, int size)
{
	char temp;
	int i;

	for (i = 0; i < size / 2; i ++) {
		temp = ptr[i];
		ptr[i] = ptr[size - 1 - i];
		ptr[size - 1 - i] = temp;
	}
}

#endif //__TRIVIAL__