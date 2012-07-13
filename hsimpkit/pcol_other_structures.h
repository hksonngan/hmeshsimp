/*
 *  Other data structures
 *		-the collapsable vertex pair
 *		-the face in vertex collapse
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

#ifndef __PCOL_OTHER__
#define __PCOL_OTHER__

#include "MixKit/MxHeap.h"
#include "util_common.h"

#define MAX_CHAR 0xff
#define FACE_INVALID MAX_CHAR


/* The pairs stored in the priority queue should always point to the valid vertices, such that the
deprecated pairs should be freed. Note that the pair_heap only stores a pointer to the real struct, 
and the 'adjacent_col_pairs' field also stores a pointer to the real 'CollapsablePair' structure
located in operating system HEAP, so 'adjacent_col_pairs' field of different vertices in fact point 
to the same struct if they are the same. This is similar to 'adjacent_faces' field which storing
the index of the face in the faces array. */
class CollapsablePair: public MxHeapable {
public:
	CollapsablePair(uint _v1, uint _v2) { set(_v1, _v2); }
	void set(uint _v1, uint _v2) { vert1 = _v1; vert2 = _v2; }

	// verts indices equals
	bool operator== (const CollapsablePair &pair) const {
		return vert1 == pair.vert1 && vert2 == pair.vert2;
	}
	// verts indices less than
	bool operator< (const CollapsablePair &pair) const {
		return vert1 < pair.vert1 || 
			vert1 == pair.vert1 && vert2 < pair.vert2;
	}
	bool operator<=  (const CollapsablePair &pair) const {
		return (*this) < pair || (*this) == pair;
	}

	bool valid() { return vert1 != vert2; }
	void keepOrder() {
		if (vert1 > vert2) 
			hswap(vert1, vert2);
	}
	void changeOneVert(uint orig, uint dst) {
		if (vert1 == orig)
			vert1 = dst;
		else if (vert2 == orig)
			vert2 = dst;
	}

	uint getAnotherVert(uint one_vert) {
		
		if (vert1 == one_vert)
			return vert2;
		if (vert2 == one_vert)
			return vert1;

		return -1;
	}

public:
	// indices of the two collapsable vertices
	uint	vert1, vert2;
	// don't know whether really need this. may
	// be a waste of memory space. may be deprecated
	HVertex	new_vertex;
};

//class QuadricPair: public CollapsablePair {
//public:
//	
//public:
//	
//};

/* Like the pairs, the vertex indices always point
   to the valid vertices, so the indices may be
   changed along the collapse. But if it is invalidated,
   it will never change. */
class CollapsableFace: public HTripleIndex<uint> {
public:
	CollapsableFace() { mark = 0; }
	void markFace(unsigned char m) { mark = m; }
	bool markIs(unsigned char m) { return mark == m; }
	void invalidate() { markFace(FACE_INVALID); }
	bool valid() { return mark != FACE_INVALID && indexValid(); }

	// this may cause the face to be invalid
	void changeOneVert(uint orig, uint dst) {
		if (!indexValid())
			return;

		if (i == orig)
			i = dst;
		else if (j == orig) 
			j = dst;
		else if (k == orig)
			k = dst;
	}

	bool indexValid() {
		return i != j && i != k && j != k;
	}

	bool indicesInRange(uint _min, uint _max) {
		return i >= _min && i <= _max &&
			j >= _min && j <= _max &&
			k >= _min && k <= _max;
	}

private:
	// face marking
	unsigned char	mark;
};

#endif //__PCOL_OTHER__