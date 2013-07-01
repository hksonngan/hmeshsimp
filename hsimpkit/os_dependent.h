/*
 *  This file defines some operating system dependent
 *  functions invoked by some of the OS independent codes, 
 *  which should be written separately on different operating
 *  systems.
 *  Currently it only contains Windows platform implementation
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef __OS_DEPENDENT__
#define __OS_DEPENDENT__

// create a file directory using the os specific kernel call.
// returns true is the directory has already existed
extern bool hCreateDir(char *dir_path);

// '\' or '/'
inline extern char* hPathSeperator();

#endif //__OS_DEPENDENT__