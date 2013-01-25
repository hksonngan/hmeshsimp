#include "ecol_iterative_quadric.h"

QuadricEdgeCollapse::QuadricEdgeCollapse() {

	// Externally visible variables
	placement_policy = MX_PLACE_OPTIMAL;
	weighting_policy = MX_WEIGHT_AREA;
	boundary_weight = INIT_BOUND_WEIGHT;
	//compactness_ratio = 0.0;
	//meshing_penalty = 1.0;
	//local_validity_threshold = 0.0;
	//vertex_degree_limit = 24;
	//will_join_only = false;
}

QuadricEdgeCollapse::~QuadricEdgeCollapse() {

}

void QuadricEdgeCollapse::allocVerts(uint _vert_count) {
	PairCollapse::allocVerts(_vert_count);
#if ARRAY_USE == ARRAY_NORMAL
	quadrics.resize(_vert_count);
#endif
}

bool QuadricEdgeCollapse::addFace(const HFace& face) {
	if (!PairCollapse::addFace(face))
		return false;

	// add the quadric matrix
	//qMatrix.calcQem(vertices[face.i], vertices[face.j], vertices[face.k]);
	float area = HFaceFormula::calcTriangleFaceArea(v(face.i), v(face.j), v(face.k));
	//qMatrix *= area;

	//quadrics[face.i] += qMatrix;
	//quadrics[face.j] += qMatrix;
	//quadrics[face.k] += qMatrix;

	/// below altered from _QSLIM_2.1_
	uint i;

	//Vec3 v1(m->vertex(f(0)));
	//Vec3 v2(m->vertex(f(1)));
	//Vec3 v3(m->vertex(f(2)));

	Vec3 v1(v(face.i).x, v(face.i).y, v(face.i).z);
	Vec3 v2(v(face.j).x, v(face.j).y, v(face.j).z);
	Vec3 v3(v(face.k).x, v(face.k).y, v(face.k).z);

	// calculating triangle plane formula
	Vec4 p = (weighting_policy == MX_WEIGHT_RAWNORMALS) ?
		triangle_raw_plane<Vec3,Vec4>(v1, v2, v3):
	triangle_plane<Vec3,Vec4>(v1, v2, v3);
	// retrieve the quadric matrix
	Quadric Q(p[X], p[Y], p[Z], p[W], area);

	switch( weighting_policy )
	{
	case MX_WEIGHT_ANGLE:
		for(i = 0; i < 3; i ++)
		{
			Quadric Q_j = Q;
			// by ht
			//Q_j *= m->compute_corner_angle(i, j);
			//quadrics(face[i]) += Q_j;
		}
		break;
	case MX_WEIGHT_AREA:
	case MX_WEIGHT_AREA_AVG:
		Q *= Q.area();
		// no break: fall through
	default:
		quadrics[face.i] += Q;
		quadrics[face.j] += Q;
		quadrics[face.k] += Q;
		break;
	}

	return true;
}

void QuadricEdgeCollapse::collectPairs() {
	int i, j;
	vert_arr starVertices;

	//for (i = 0; i < vertices.count(); i ++) {

	//	collectStarVertices(i, &starVertices);
	//	for (j = 0; j < starVertices.count(); j ++)
	//		// add specific edge only once
	//		// never collapse the exterior vertices
	//		if (i < starVertices[j] && !vertices[i].exterior() && !vertices[starVertices[j]].exterior()) {

	//			CollapsablePair *new_pair = new CollapsablePair(i, starVertices[j]);
	//			evaluatePair(new_pair);
	//			addCollapsablePair(new_pair);

	//			// add boundary constraint
	//			if (boundary_weight > 0) {
	//				collectEdgeFaces(i, starVertices[j], _faces);
	//				// is the boundary edge
	//				if (_faces.count() == 1)
	//					addDiscontinuityConstraint(i, starVertices[j], _faces[0]);
	//			}
	//		}
	//}

	_for_loop(vertices, ECVertexMap) {
		i = _retrieve_index();
		collectStarVertices(i, &starVertices);
		for (j = 0; j < starVertices.count(); j ++)
			// add specific edge only once
			// never collapse the exterior vertices
			if (i < starVertices[j] && !v(i).exterior() && !v(starVertices[j]).exterior()) {
				CollapsablePair *new_pair = new CollapsablePair(i, starVertices[j]);
				addCollapsablePair(new_pair);
			}
	}
}

void QuadricEdgeCollapse::addCollapsablePair(CollapsablePair *new_pair) {
	face_arr _faces;

	evaluatePair(new_pair);
	PairCollapse::addCollapsablePair(new_pair);

	// add boundary constraint
	if (boundary_weight > 0) {
		collectEdgeFaces(new_pair->vert1, new_pair->vert2, _faces);
		// is the boundary edge
		if (_faces.count() == 1)
			addDiscontinuityConstraint(new_pair->vert1, new_pair->vert2, _faces[0]);
	}
}

void QuadricEdgeCollapse::initialize() {
	PairCollapse::initialize();
	ostringstream ostr;
#if ARRAY_USE == ARRAY_USE_HASH
	ostr << "\tquadrics buckets: " << quadrics.bucket_count() << endl
		<< "\tavg num of quadrics per bucket: " << quadrics.load_factor() << endl
		<< "\tmax load factor: " << quadrics.max_load_factor() << endl << endl;
	addInfo(ostr.str());
#endif
}

HVertex QuadricEdgeCollapse::evaluatePair(CollapsablePair *pair) {
	// below altered from _QSLIM_2.1_
	const Quadric &Qi = q(pair->vert1), &Qj = q(pair->vert2);
	Quadric Q = Qi;  Q += Qj;
	double e_min;

	// the matrix is not singular
	if( placement_policy == MX_PLACE_OPTIMAL &&
		Q.optimize(&pair->new_vertex.x, 
			&pair->new_vertex.y, 
			&pair->new_vertex.z) ) {
			e_min = Q.evaluate(pair->new_vertex.x, 
						pair->new_vertex.y, pair->new_vertex.z);
	} else {
		Vec3 vi(v(pair->vert1).x, v(pair->vert1).y, v(pair->vert1).z), 
			vj(v(pair->vert2).x, v(pair->vert2).y, v(pair->vert2).z);
		Vec3 best;

		// evaluate along the line formed by vi, vj
		if( placement_policy >= MX_PLACE_LINE && Q.optimize(best, vi, vj) )
			e_min = Q(best);
		else {
			double ei = Q(vi), ej = Q(vj);
			if( ei < ej ) { e_min = ei; best = vi; }
			else          { e_min = ej; best = vj; }

			// evaluate the mid point
			if( placement_policy >= MX_PLACE_ENDORMID ) {
				Vec3 mid = (vi + vj) / 2.0;
				double e_mid = Q(mid);
				if(e_mid < e_min) { 
					e_min = e_mid; 
					best = mid; 
				}
			}
		}

		pair->new_vertex.x = best[X];
		pair->new_vertex.y = best[Y];
		pair->new_vertex.z = best[Z];
	}

	if (weighting_policy == MX_WEIGHT_AREA_AVG)
		e_min /= Q.area();

	// note this~ it set the key
	pair->heap_key(-e_min);
	return pair->new_vertex;
}

void QuadricEdgeCollapse::collapsePair(pCollapsablePair pair) {
	uint vert1 = pair->vert1, vert2 = pair->vert2;
	q(vert1) += q(vert2);
	PairCollapse::collapsePair(pair);
#if ARRAY_USE == ARRAY_USE_HASH
	quadrics.erase(vert2);
#endif
}