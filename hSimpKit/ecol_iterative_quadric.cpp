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
	
}

//bool QuadricEdgeCollapse::targetVert(uint target_count) {
//
//	CollapsablePair *pair;
//
//	while(valid_verts > target_count) {
//		
//		pair = extractTopPair();
//	}
//
//	return true;
//}

HVertex QuadricEdgeCollapse::evaluatePair(CollapsablePair *pair) {

	//qMatrix = quadrics[pair->vert1];
	//qMatrix += quadrics[pair->vert2];

	// the matrix is not singular
	//if (qMatrix.calcRepresentativeVertex(pair->new_vertex)) {
	//	return pair->new_vertex;
	//}

	/// below altered from _QSLIM_2.1_

	const Quadric &Qi = quadrics[pair->vert1], &Qj=quadrics[pair->vert2];

	Quadric Q = Qi;  Q += Qj;
	double e_min;

	// the matrix is not singular
	if( placement_policy == MX_PLACE_OPTIMAL &&
		Q.optimize(&pair->new_vertex.x, &pair->new_vertex.y, &pair->new_vertex.y) ) {

			e_min = Q.evaluate(pair->new_vertex.x, pair->new_vertex.y, pair->new_vertex.z);
	}
	else {

		Vec3 vi(vertices[pair->vert1].x, vertices[pair->vert1].y, vertices[pair->vert1].z), 
			vj(vertices[pair->vert2].x, vertices[pair->vert2].y, vertices[pair->vert2].z);
		Vec3 best;

		// evaluate along the line formed by vi, vj
		if( placement_policy >= MX_PLACE_LINE && Q.optimize(best, vi, vj) )
			e_min = Q(best);
		else {

			double ei = Q(vi), ej = Q(vj);

			if( ei < ej ) { e_min = ei; best = vi; }
			else          { e_min = ej; best = vj; }

			// evaluate the mid point
			if( placement_policy >= MX_PLACE_ENDORMID )
			{
				Vec3 mid = (vi + vj) / 2.0;
				double e_mid = Q(mid);

				if( e_mid < e_min ) { e_min = e_mid; best = mid; }
			}
		}

		pair->new_vertex.x = best[X];
		pair->new_vertex.y = best[Y];
		pair->new_vertex.z = best[Z];
	}

	if( weighting_policy == MX_WEIGHT_AREA_AVG )
		e_min /= Q.area();

	// note this~ it set the key
	pair->heap_key(- e_min);

	return pair->new_vertex;
}

void QuadricEdgeCollapse::collapsePair(pCollapsablePair &pair) {

	quadrics[pair->vert1] += quadrics[pair->vert2];
	PairCollapse::collapsePair(pair);
}