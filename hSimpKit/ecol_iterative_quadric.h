/*
 *  iterative edge collapse using quadric error metrics
 *  refer to THE PAPER (you must know it)
 *
 *  author: ht
 */

#ifndef __ITERATIVE_QUADRIC_EDGE_COLLAPSE__
#define __ITERATIVE_QUADRIC_EDGE_COLLAPSE__

#include "pcol_iterative.h"

class QuadricEdgeCollapse: public PairCollapse {
public:
	void allocVerts(uint _vert_count);
	inline void addFace(HTripleIndex face);

protected:
	HDynamArray<q_matrix> quadrics;

	static q_matrix	qMatrix;
};

void QuadricEdgeCollapse::addFace(HTripleIndex face) {

	PairCollapse::addFace(face);

	// add the quadric matrix
	qMatrix.calcQem(vertices[face.i], vertices[face.j], vertices[face.k]);
	float area = HFaceFormula::calcTriangleFaceArea(vertices[face.i], vertices[face.j], vertices[face.k]);
	qMatrix *= area;

	quadrics[face.i] += qMatrix;
	quadrics[face.j] += qMatrix;
	quadrics[face.k] += qMatrix;
}

#endif //__ITERATIVE_QUADRIC_EDGE_COLLAPSE__