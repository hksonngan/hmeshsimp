#include "trivial.h"

void trimExtAndAppend(char *src, char *dst, char *app)
{
	string s(src);
	s = s.substr(0, s.find_last_of("."));
	s += app;

	memcpy(dst, s.c_str(), s.size());
	dst[s.size()] = '\0';
}

EndianOrder getSystemEndianMode()
{
	unsigned short test = 0x1122;

	if( *( (unsigned char*) &test ) == 0x11 )
		return H_BIG_ENDIAN;
	else
		return H_LITTLE_ENDIAN;
}
