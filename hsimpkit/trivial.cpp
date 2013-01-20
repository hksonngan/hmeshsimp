#include "trivial.h"
#include <limits>

void trimExtAndAppend(char *src, char *dst, char *app)
{
	string s(src);
	s = s.substr(0, s.find_last_of("."));
	s += app;

	memcpy(dst, s.c_str(), s.size());
	dst[s.size()] = '\0';
}

string getFilename(const char *filepath)
{
	string filename = getExtFilename(filepath);

	int end;

	end = filename.find_last_of('.');
	if (end == string::npos) {
		end = filename.size();
	}

	return filename.substr(0, end);
}

string getExtFilename(const char *filepath) {
	string filename(filepath);

	int index1 = filename.find_last_of('\\');
	int index2 = filename.find_last_of('/');
	int start;

	if (index1 == string::npos && index2 != string::npos) {
		start = index2 + 1;
	}
	else if (index1 != string::npos && index2 == string::npos) {
		start = index1 + 1;
	}
	else if (index1 != string::npos && index2 != string::npos) {
		start = std::max(index1, index2) + 1;
	}
	else {
		start = 0;
	}

	return filename.substr(start);
}

string getFileExtension(const char *filepath) {
	string filename(filepath);
	int i = filename.find_last_of('.');
	return filename.substr(i + 1);
}

EndianOrder getSystemEndianMode() {
	unsigned short test = 0x1122;

	if( *( (unsigned char*) &test ) == 0x11 )
		return H_BIG_ENDIAN;
	else
		return H_LITTLE_ENDIAN;
}

char* getTime() {
	time_t rawtime;
	struct tm * timeinfo;
	time (&rawtime);
	timeinfo = localtime (&rawtime);

	return asctime(timeinfo);
}

void stringToCstr(string &str, char* cstr)
{
	memcpy(cstr, str.c_str(), str.size() * sizeof(char));
	cstr[str.size()] = '\0';
}

char* getPlyBinaryFormat() {

	EndianOrder e = getSystemEndianMode();
	if (H_BIG_ENDIAN)
		return "binary_big_endian 1.0";
	else
		return "binary_little_endian 1.0";
}
