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
extern void trimExtAndAppend(char *src, char *dst, char *app);

// get file name without extensions
extern string getFilename(char *filepath);

// get the system endian mode
extern EndianOrder getSystemEndianMode();

inline void stringToCstr(string &str, char* cstr)
{
	memcpy(cstr, str.c_str(), str.size() * sizeof(char));
	cstr[str.size()] = '\0';
}

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