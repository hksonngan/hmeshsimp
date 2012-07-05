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
	// indices of the two collapsable vertices
	uint	vert1, vert2;

public:
	float	getError();
};

class CollapseFace: public HTripleIndex<uint> {
public:
	void	invalidate() { _valid = false; }
	bool	valid() { return _valid; }

private:
	bool	_valid;
};

#endif //__PCOL_OTHER__