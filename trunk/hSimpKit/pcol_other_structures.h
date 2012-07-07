/*
 *  other data structures
 *		-the collapsable vertex pair
 *		-the face in vertex collapse
 *
 *  author: ht
 */

#ifndef __PCOL_OTHER__
#define __PCOL_OTHER__

#include "MxHeap.h"
#include "util_common.h"

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

	bool operator== (const CollapsablePair &pair) const {
		return vert1 == pair.vert1 && vert2 == pair.vert2;
	}
	bool operator< (const CollapsablePair &pair) const {
		return vert1 < pair.vert1 || 
			vert1 == pair.vert1 && vert2 < pair.vert2;
	}
	bool operator<=  (const CollapsablePair &pair) const {
		return (*this) < pair || (*this) == pair;
	}

	bool valid() { return vert1 != vert2; }
	bool keepOrder() {
		if (vert1 > vert2) 
			hswap(vert1, vert2);
	}
	void changeOneVert(uint orig, uint dst) {
		if (vert1 == orig)
			vert1 = dst;
		else if (vert2 == orig)
			vert2 = dst;
	}

public:
	// indices of the two collapsable vertices
	uint	vert1, vert2;
	// don't know whether really need this. may
	// be a waste of memory space. may be deprecated
	HVertex	new_vertex;
};

class QuadricPair: public CollapsablePair {
public:
	
public:
	
};

class CollapseFace: public HTripleIndex<uint> {
public:
	void	invalidate() { _valid = false; }
	bool	valid() { return _valid; }

private:
	bool	_valid;
};

#endif //__PCOL_OTHER__