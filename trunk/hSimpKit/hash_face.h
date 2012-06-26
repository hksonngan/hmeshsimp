/*
 *  all about hash sets containing degenerated faces
 *
 *  author : ht
 *  email  : waytofall916@gmail.com
 */

#ifndef __HASH_FACE__
#define __HASH_FACE__

#include "util_common.h"
#include <boost/unordered_set.hpp>


/* -- the face index (three uniform cluster index, 
      thus three HTripleIndex which is three integer)
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
class HFaceIndexEqual
{
public:
	bool operator()(const HFaceIndex& h1, const HFaceIndex& h2) const {
		return h1 == h2;
	}
};

/* type of degenerated face container */
typedef boost::unordered::unordered_set<HFaceIndex, HFaceIndexHash, HFaceIndexEqual> HFaceIndexSet; // HDegFaceContainer deprecated

/* -- the HTripleIndex hash definitions -- */

/* -- hash -- */

/* the hash functor */
class HTripleIndexHash
{
public:
	size_t operator()(const HTripleIndex<Integer>& index) const {
		unsigned long h = 0;
		Integer arr[3];

		index.sortIndex(arr);

		h += arr[0] & 0x000003ff; h <<= 10;
		h += arr[1] & 0x000003ff; h <<= 10;
		h += arr[2] & 0x000003ff;

		return size_t(h);
	}
};

/* the equal functor */
class HTripleIndexEqual
{
public:
	bool operator()(const HTripleIndex<Integer>& h1, const HTripleIndex<Integer>& h2) const {
		return h1.unsequncedEqual(h2);
	}
};

/* type of degenerated face container */
typedef boost::unordered::unordered_set<HTripleIndex<Integer>, HTripleIndexHash, HTripleIndexEqual> HTripleIndexSet;

#endif //__HASH_FACE__