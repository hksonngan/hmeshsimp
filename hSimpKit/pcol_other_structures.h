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

class CollapsablePair: public MxHeapable {
public:
	CollapsablePair(uint _v1, uint _v2) { set(_v1, _v2); }
	void	set(uint _v1, uint _v2) { vert1 = _v1; vert2 = _v2; }

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