/*
 *  include file for transplant of Garland's code
 *
 *  author: ht
 */

#ifndef __H_USE_MIX__
#define __H_USE_MIX__

#include "compiler_config.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef UINT_MAX
#  define MXID_NIL UINT_MAX
#else
#  define MXID_NIL 0xffffffffU
#endif

#if !defined(HAVE_UINT) && !defined(uint)
typedef unsigned int uint;
#endif

#if !defined(HAVE_USHORT) && !defined(ushort)
typedef unsigned short ushort;
#endif

#define MIN(a,b) (((a)>(b))?(b):(a))
#define MAX(a,b) (((a)>(b))?(a):(b))

#ifndef ABS
#  define ABS(x) (((x)<0)?-(x):(x))
#endif

#ifndef MIX_NO_AXIS_NAMES
enum Axis {X=0, Y=1, Z=2, W=3};
#endif

#define SanityCheck(t)
#define PARANOID(x)

#define AssertBound(t)
#define PRECAUTION(x)

#endif //__H_USE_MIX__
