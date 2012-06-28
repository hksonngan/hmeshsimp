/*
 *  this file defines some operating system dependent
 *  functions invoked by some of the os independent codes, 
 *  which should be written separately on different operating
 *  systems.
 *
 *  ht waytofall916@gmail.com
 */

#ifndef __OS_DEPENDENT__
#define __OS_DEPENDENT__

// create a file directory using the os specific kernel call.
// returns true is the directory has already existed
extern bool createDir(char *dir_path);

#endif //__OS_DEPENDENT__