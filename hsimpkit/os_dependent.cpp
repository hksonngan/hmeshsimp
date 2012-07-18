/*
 *  this is a implementation on Windows platform
 *  you should implement it on your own on different
 *  platform, like say, a UNIX or Linux
 */

#include "os_dependent.h"
#include "windows.h"

bool hCreateDir(char *dir_path)
{
	if (CreateDirectory(dir_path, NULL))
		return true;

	if (GetLastError() == ERROR_ALREADY_EXISTS)
		return true;

	return false;
}

inline extern char* hPathSeperator()
{
	return "/";
}