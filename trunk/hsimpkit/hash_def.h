/*
 *  All about hash sets containing degenerated faces
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __HASH_DEF__
#define __HASH_DEF__

#include "common_types.h"
#include <boost/unordered_set.hpp>

using boost::unordered::unordered_set;


/* -- the face index (three uniform cluster index, 
      thus three HTriple which is three integer)
	  hash definition -- */

/* the hash functor */
class HFaceIndexHash
{
public:
	size_t operator()(const HFaceIndex& index) const {
		unsigned long h = 0;

		h += index.v1CIndex.i & 0x0000000f; h <<= 4;
		h += index.v1CIndex.j & 0x0000000f; h <<= 4;
		h += index.v1CIndex.k & 0x0000000f; h <<= 4;
		h += index.v2CIndex.i & 0x0000000f; h <<= 4;
		h += index.v2CIndex.j & 0x0000000f; h <<= 4;
		h += index.v2CIndex.k & 0x0000000f; h <<= 4;
		h += index.v3CIndex.i & 0x0000000f; h <<= 4;
		h += index.v3CIndex.j & 0x0000000f;
		h += index.v3CIndex.k;

		return size_t(h);
	}
};

/* the equal functor */
/* !! this should be the unsequenced equal !! */
class HFaceIndexEqual
{
public:
	bool operator()(const HFaceIndex& h1, const HFaceIndex& h2) const {
		return h1 == h2;
	}
};

/* type of degenerated face container for vertex clustering */
typedef unordered_set<HFaceIndex, HFaceIndexHash, HFaceIndexEqual> HFaceIndexSet;

/* -- the HTriple<uint> hash definitions -- */

/* -- hash -- */

/* the hash functor */
/* !!this name is a little improper, will change it or not */ 
class HTripleHash
{
public:
	size_t operator()(const HTriple<uint>& index) const {
		unsigned long h = 0;
		uint arr[3];

		index._sortIndex(arr);

		h += arr[0] & 0x000003ff; h <<= 10;
		h += arr[1] & 0x000003ff; h <<= 10;
		h += arr[2] & 0x000003ff;

		return size_t(h);
	}
};

/* the unsequenced equal functor (which means <1,2,3> = <2,1,3>) */
class HTripleEqual
{
public:
	bool operator()(const HTriple<uint>& h1, const HTriple<uint>& h2) const {
		return h1.unsequencedEqual(h2);
	}
};

/* type of degenerated face container for spatial division */
typedef unordered_set<HTriple<uint>, HTripleHash, HTripleEqual> HTripleSet;

/* the sequenced equal functor (which means <1,2,3> != <2,1,3>) */
class HTripleSequencedEqual
{
public:
	bool operator()(const HTriple<uint>& h1, const HTriple<uint>& h2) const {
		return h1 == h2;
	}
};

#endif //__HASH_DEF__