/*
 *  Iterative edge collapse using quadric error metrics
 *  You can refer to THE PAPER (you must have known it)
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __ITERATIVE_QUADRIC_EDGE_COLLAPSE__
#define __ITERATIVE_QUADRIC_EDGE_COLLAPSE__

#include "pcol_iterative.h"
#include "MixKit/MxQMetric3.h"
// for some constants and typedefs
//#include "MxStdSlim.h"
#include "gfx/geom3d.h"

/// definitions from _QSLIM_

#define MX_PLACE_ENDPOINTS 0
#define MX_PLACE_ENDORMID  1
#define MX_PLACE_LINE      2
#define MX_PLACE_OPTIMAL   3

#define MX_WEIGHT_UNIFORM       0
#define MX_WEIGHT_AREA          1
#define MX_WEIGHT_ANGLE         2
#define MX_WEIGHT_AVERAGE       3
#define MX_WEIGHT_AREA_AVG      4
#define MX_WEIGHT_RAWNORMALS    5

#define INIT_BOUND_WEIGHT 10000.0f

typedef MxQuadric3 Quadric;

class QuadricEdgeCollapse: public PairCollapse {

public:

	////////////////////////////////////////
	// initializers
	////////////////////////////////////////

	QuadricEdgeCollapse();
	~QuadricEdgeCollapse();
	virtual void allocVerts(uint _vert_count);
	virtual bool addFace(HFace face);
	virtual void collectPairs();
	// init after the vertices and faces are ready
	virtual void initialize();
	inline void addDiscontinuityConstraint(uint vert1, uint vert2, uint face_id);


	////////////////////////////////////////
	// computing
	////////////////////////////////////////

	// evaluate the target placement and error incurred,
	// and update the pair's content
	virtual HVertex evaluatePair(CollapsablePair *pair);
	virtual void collapsePair(pCollapsablePair &pair);

protected:
	// every vertex has a quadric
	HDynamArray<q_matrix> quadrics;

public:

	////////////////////////////////////////
	// configuration variables
	////////////////////////////////////////

	int weighting_policy;
	int placement_policy;
	double boundary_weight;
};

void QuadricEdgeCollapse::addDiscontinuityConstraint(uint vert1, uint vert2, uint face_id) {

	Vec3 org(vertices[vert1].x, vertices[vert1].y, vertices[vert1].z), 
		dest(vertices[vert2].x, vertices[vert2].y, vertices[vert2].z);
	Vec3 e = dest - org;

	Vec3 n;
	HNormal hn;
	CollapsableFace &f = faces[face_id];
	HFaceFormula::calcFaceNormal(vertices[f.i], vertices[f.j], vertices[f.k], hn);
	assign(n, hn);

	Vec3 n2 = e ^ n;
	unitize(n2);

	MxQuadric3 Q(n2, - (n2 * org));
	Q *= boundary_weight;

	if( weighting_policy == MX_WEIGHT_AREA ||
	    weighting_policy == MX_WEIGHT_AREA_AVG )
	{
	    Q.set_area(norm2(e));
	    Q *= Q.area();
	}

	quadrics[vert1] += Q;
	quadrics[vert2] += Q;
}

#endif //__ITERATIVE_QUADRIC_EDGE_COLLAPSE__