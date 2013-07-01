/*
 *  Some Trivial But Very Commonly Used Operations
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef __TRIVIAL__
#define __TRIVIAL__

#include <string>
#include <time.h>

using std::string;

// specify the endian order
enum EndianOrder { H_BIG_ENDIAN, H_LITTLE_ENDIAN };

// trim the extention of the file name and append some str
void trimExtAndAppend(char *src, char *dst, char *app);

// get file name without extension and directory path
string getFilename(const char *filepath);

// get file name with extension
string getExtFilename(const char *filepath);

string getFileExtension(const char *filepath);

string getNextWordBetweenSpace(const int pos, const string& str);

// get the system endian mode
EndianOrder getSystemEndianMode();

char* getTime();

void stringToCstr(string &str, char* cstr);

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

char* getPlyBinaryFormat();

#endif //__TRIVIAL__