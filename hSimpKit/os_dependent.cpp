/*
 *  this is a implementation on Windows platform
 *  you should implement it on your own on different
 *  platform, like say, a UNIX or Linux
 */

#include "os_dependent.h"
#include "windows.h"

bool hCreateDir(char *dir_path)
{
	CreateDirectory(dir_path, NULL);

	return true;
}

inline extern char* hPathSeperator()
{
	return "/";
}