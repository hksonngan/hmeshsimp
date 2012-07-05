#include "ecol_iterative_quadric.h"

void QuadricEdgeCollapse::allocVerts(uint _vert_count) {

	PairCollapse::allocVerts(_vert_count);
	quadrics.resize(_vert_count);
}