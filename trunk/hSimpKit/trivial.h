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

// trim the extention of the file name and append some str
inline void trimExtAndAppend(char *src, char *dst, char *app)
{
	string s(src);
	s = s.substr(0, s.find_last_of("."));
	s += app;

	memcpy(dst, s.c_str(), s.size());
	dst[s.size()] = '\0';
}

#endif //__TRIVIAL__