/*
 *  All about hash sets containing degenerated faces
 *
 *  Author: Ht
 *  Email : waytofall916@gmail.com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 *	
 *  This file is part of hmeshsimp.
 *
 *  hmeshsimp is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  hmeshsimp is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with hmeshsimp.  If not, see <http://www.gnu.org/licenses/>.
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

/* type of degenerated face container for vertex clustering */
typedef boost::unordered::unordered_set<HFaceIndex, HFaceIndexHash, HFaceIndexEqual> HFaceIndexSet;

/* -- the HTripleIndex hash definitions -- */

/* -- hash -- */

/* the hash functor */
class HTripleIndexHash
{
public:
	size_t operator()(const HTripleIndex<uint>& index) const {
		unsigned long h = 0;
		uint arr[3];

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
	bool operator()(const HTripleIndex<uint>& h1, const HTripleIndex<uint>& h2) const {
		return h1.unsequencedEqual(h2);
	}
};

/* type of degenerated face container for spatial division */
typedef boost::unordered::unordered_set<HTripleIndex<uint>, HTripleIndexHash, HTripleIndexEqual> HTripleIndexSet;

#endif //__HASH_FACE__