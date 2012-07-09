/*
 *  iterative edge collapse using quadric error metrics
 *  refer to THE PAPER (you must know it)
 *
 *  author: ht
 */

#ifndef __ITERATIVE_QUADRIC_EDGE_COLLAPSE__
#define __ITERATIVE_QUADRIC_EDGE_COLLAPSE__

#include "pcol_iterative.h"
#include "MxQMetric3.h"
// for some constants and typedefs
#include "MxStdSlim.h"
#include "gfx/geom3d.h"

typedef MxQuadric3 Quadric;

class QuadricEdgeCollapse: public PairCollapse {
public:
	////////////////////////////////////////
	// initializers
	////////////////////////////////////////

	QuadricEdgeCollapse();
	virtual void allocVerts(uint _vert_count);
	virtual bool addFace(HFace face);
	virtual void collectPairs();
	// init after the vertices and faces are ready
	virtual void initialize();

	////////////////////////////////////////
	// computing
	////////////////////////////////////////

	// simplify targeting vertex
	//bool targetVert(uint target_count);

	// evaluate the target placement and error incurred,
	// and update the pair's content
	virtual HVertex evaluatePair(CollapsablePair *pair);
	virtual void collapsePair(pCollapsablePair &pair);

protected:
	// every vertex has a quadric
	HDynamArray<q_matrix> quadrics;

	////////////////////////////////////////
	// configuration variables
	////////////////////////////////////////
	int weighting_policy;
	int placement_policy;

	//static q_matrix	qMatrix;
};

#endif //__ITERATIVE_QUADRIC_EDGE_COLLAPSE__