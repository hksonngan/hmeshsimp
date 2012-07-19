/*
 *  Other data structures
 *		-the collapsable vertex pair
 *		-the face in vertex collapse
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 */

#ifndef __PCOL_OTHER__
#define __PCOL_OTHER__

#include "MixKit/MxHeap.h"
#include "util_common.h"

#define MAX_CHAR 0xff
#define FACE_INVALID MAX_CHAR


/* The pairs stored in the priority queue should always point to the valid vertices, such that the
abandoned pairs should be freed. Note that the pair_heap only stores a pointer to the real struct, 
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
	// be a waste of memory space. may be abandoned
	HVertex	new_vertex;
};

/* Like the pairs, the vertex indices always point
   to the valid vertices, so the indices may be
   changed along the collapse. But if it is invalidated,
   it will never change. */
class CollapsableFace: public HTripleIndex<uint> {
public:
	CollapsableFace() { mark = 0; }
	void markFace(uchar m) { mark = m; }
	bool markIs(uchar m) { return mark == m; }
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
	uchar	mark;
};

#endif //__PCOL_OTHER__