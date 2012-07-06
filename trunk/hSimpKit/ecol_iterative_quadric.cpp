#include "ecol_iterative_quadric.h"

QuadricEdgeCollapse::QuadricEdgeCollapse() {

	// Externally visible variables
	placement_policy = MX_PLACE_OPTIMAL;
	weighting_policy = MX_WEIGHT_AREA;
	//boundary_weight = 1000.0;
	//compactness_ratio = 0.0;
	//meshing_penalty = 1.0;
	//local_validity_threshold = 0.0;
	//vertex_degree_limit = 24;
	//will_join_only = false;
}

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
			// add specific edge only once
			if (i < starVertices[j]) {
				CollapsablePair *new_pair = new CollapsablePair(i, starVertices[j]);
				evaluatePair(new_pair);
				addCollapsablePair(new_pair);
			}
	}
}

void QuadricEdgeCollapse::initialize() {

	PairCollapse::intialize();
	collectPairs();
}

bool QuadricEdgeCollapse::targetVert(uint target_count) {

	CollapsablePair *pair;

	while(valid_verts > target_count) {
		
		pair = extractTopPair();
	}
}