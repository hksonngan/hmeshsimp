/*
 *  Common I/O Routines
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */

#ifndef __H_IO_COMMON__
#define __H_IO_COMMON__

#include <iostream>
#include "common_def.h"

using std::ostream;
using std::istream;
using std::endl;


#define WRITE_PATCH_BINARY

#define WRITE_BLOCK_BIN(out, n, size)		out.write((char *)&n, size)
#define WRITE_UINT_BIN(out, n)				out.write((char *)&n, sizeof(uint))

#define READ_BLOCK_BIN(in, n, size)			in.read((char *)&n, size)
#define READ_UINT_BIN(in, n)				in.read((char *)&n, sizeof(uint))

#ifdef WRITE_PATCH_BINARY
	#define WRITE_BLOCK(out, n, size)		WRITE_BLOCK_BIN(out, n, size)
	#define WRITE_UINT(out, n)				WRITE_UINT_BIN(out, n)

	#define READ_BLOCK(in, n, size)			READ_BLOCK_BIN(in, n, size)
	#define READ_UINT(in, n)				READ_UINT_BIN(in, n)
#else
	#define WRITE_BLOCK(out, n, size)		out << n << " "
	#define WRITE_UINT(out, n)				out << n << " "

	#define READ_BLOCK(in, n, size)			in >> n
	#define READ_UINT(in, n)				in >> n 
#endif

#define C_READ_BLOCK(fp, n, size, count)	fread((void *)&n, size, count, fp)
#define C_WRITE_BLOCK(fp, n, size, count)	fwrite((void *)&n, size, count, fp)

#define VERT_ITEM_SIZE sizeof(float)

#endif //__H_IO_COMMON__