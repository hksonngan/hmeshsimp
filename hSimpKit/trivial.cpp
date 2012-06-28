#include "trivial.h"

void trimExtAndAppend(char *src, char *dst, char *app)
{
	string s(src);
	s = s.substr(0, s.find_last_of("."));
	s += app;

	memcpy(dst, s.c_str(), s.size());
	dst[s.size()] = '\0';
}

string getFilename(char *filepath)
{
	string filename(filepath);

	int index1 = filename.find_last_of('\\');
	int index2 = filename.find_last_of('/');
	int start, end;

	if (index1 == string::npos && index2 != string::npos) {
		start = index2 + 1;
	}
	else if (index1 != string::npos && index2 == string::npos) {
		start = index1 + 1;
	}
	else if (index1 != string::npos && index2 != string::npos) {
		start = max(index1, index2) + 1;
	}
	else {
		start = 0;
	}

	end = filename.find_last_of('.');
	if (end == string::npos) {
		end = filename.size();
	}

	return filename.substr(start, end - start);
}

EndianOrder getSystemEndianMode()
{
	unsigned short test = 0x1122;

	if( *( (unsigned char*) &test ) == 0x11 )
		return H_BIG_ENDIAN;
	else
		return H_LITTLE_ENDIAN;
}
