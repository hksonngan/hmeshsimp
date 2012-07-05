#include "ecol_iterative_quadric.h"

void QuadricEdgeCollapse::allocVerts(uint _vert_count) {

	PairCollapse::allocVerts(_vert_count);
	quadrics.resize(_vert_count);
}

void QuadricEdgeCollapse::collectPairs() {

	int i, j;
	vert_arr starVertices;
	CollapsablePair *pair;

	for (i = 0; i < vertices.count(); i ++) {

		collectStarVertices(i, &starVertices);
		for (j = 0; j < starVertices.count(); j ++)
			if (i < starVertices[j]) {
				pair = createPair(i, starVertices[j]);
				
			}
	}
}