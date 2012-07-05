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
	////////////////////////////////////////
	// initializers
	////////////////////////////////////////
	void allocVerts(uint _vert_count);
	inline void addFace(HTripleIndex face);
	void collectPairs();
	inline CollapsablePair* createPair(uint _vert1, uint _vert2);

	///////////////////////////////////////
	// computing
	///////////////////////////////////////
	inline HVertex optimizeNewVertex(CollapsablePair *pair);

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

CollapsablePair* QuadricEdgeCollapse::createPair(uint _vert1, uint _vert2) {
	
	CollapsablePair *new_pair = new CollapsablePair(_vert1, _vert2);
	
}

inline HVertex QuadricEdgeCollapse::optimizeNewVertex(CollapsablePair *pair) {

	qMatrix = quadrics[pair->vert1];
	qMatrix += quadrics[pair->vert2];

	// the matrix is not singular
	if (qMatrix.calcRepresentativeVertex(pair->new_vertex)) {
		return pair->new_vertex;
	}

	
}

#endif //__ITERATIVE_QUADRIC_EDGE_COLLAPSE__