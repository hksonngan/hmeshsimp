/*
 *  Iterative edge collapse using quadric error metrics
 *  You can refer to THE PAPER (you must have known it)
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */


#ifndef __ITERATIVE_QUADRIC_EDGE_COLLAPSE__
#define __ITERATIVE_QUADRIC_EDGE_COLLAPSE__

#include "pcol_iterative.h"
#include "MixKit/MxQMetric3.h"
// for some constants and typedefs
//#include "MxStdSlim.h"
#include "gfx/geom3d.h"
#include "h_math.h"


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

#define INIT_BOUND_WEIGHT	10000.0f

typedef MxQuadric3 Quadric;
typedef unordered_map<uint, Quadric> QuadMap;

class QuadricEdgeCollapse: public PairCollapse {

public:

	////////////////////////////////////////
	// initializers
	////////////////////////////////////////

	QuadricEdgeCollapse();
	~QuadricEdgeCollapse();
	virtual void allocVerts(uint _vert_count);
	inline void addFaceQuadric(const HFace& face);
	virtual bool addFace(const HFace& face);
	// used for incremental simplification
	virtual bool addFace(const uint & index, const HFace& face);
	virtual void addCollapsablePair(CollapsablePair *new_pair);
	virtual void collectPairs();
	// init after the vertices and faces are ready
	virtual void initialize();
	inline void addDiscontinuityConstraint(uint vert1, uint vert2, uint face_id);
	Quadric& q(uint i) { return quadrics.at(i); }
	inline uint quadCount();


	////////////////////////////////////////
	// computing
	////////////////////////////////////////

	// evaluate the target placement and error incurred,
	// and update the pair's content
	virtual HVertex evaluatePair(CollapsablePair *pair);
	virtual void collapsePair(pCollapsablePair pair);

protected:
	// every vertex has a quadric
#if ARRAY_USE == ARRAY_NORMAL
	HDynamArray<Quadric> quadrics;
#else
	QuadMap quadrics;
#endif

public:

	////////////////////////////////////////
	// configuration variables
	////////////////////////////////////////

	int weighting_policy;
	int placement_policy;
	double boundary_weight;
};

void QuadricEdgeCollapse::addFaceQuadric(const HFace& face) {
	// add the quadric matrix
	float area = HFaceFormula::calcTriangleFaceArea(v(face.i), v(face.j), v(face.k));

	// below altered from _QSLIM_2.1_
	Vec3 v1(v(face.i).x, v(face.i).y, v(face.i).z);
	Vec3 v2(v(face.j).x, v(face.j).y, v(face.j).z);
	Vec3 v3(v(face.k).x, v(face.k).y, v(face.k).z);

	// calculating triangle plane formula
	Vec4 p = (weighting_policy == MX_WEIGHT_RAWNORMALS) ?
		triangle_raw_plane<Vec3,Vec4>(v1, v2, v3):
	triangle_plane<Vec3,Vec4>(v1, v2, v3);
	// retrieve the quadric matrix
	Quadric Q(p[X], p[Y], p[Z], p[W], area);

	Q *= Q.area();
	quadrics[face.i] += Q;
	quadrics[face.j] += Q;
	quadrics[face.k] += Q;

	//switch( weighting_policy )
	//{
	//case MX_WEIGHT_ANGLE:
	//	for(uint i = 0; i < 3; i ++)
	//	{
	//		Quadric Q_j = Q;
	//		// by ht
	//		//Q_j *= m->compute_corner_angle(i, j);
	//		//quadrics(face[i]) += Q_j;
	//	}
	//	break;
	//case MX_WEIGHT_AREA:
	//case MX_WEIGHT_AREA_AVG:
	//	Q *= Q.area();
	//	// no break: fall through
	//default:
	//	quadrics[face.i] += Q;
	//	quadrics[face.j] += Q;
	//	quadrics[face.k] += Q;
	//	break;
	//}
}

void QuadricEdgeCollapse::addDiscontinuityConstraint(uint vert1, uint vert2, uint face_id) {

	Vec3 org(v(vert1).x, v(vert1).y, v(vert1).z), 
		dest(v(vert2).x, v(vert2).y, v(vert2).z);
	Vec3 e = dest - org;

	Vec3 n;
	HNormal hn;
	CollapsableFace &face = f(face_id);
	HFaceFormula::calcFaceNormal(v(face.i), v(face.j), v(face.k), hn);
	assign(n, hn);

	Vec3 n2 = e ^ n;
	unitize(n2);

	MxQuadric3 Q(n2, - (n2 * org));
	Q *= boundary_weight;

	if( weighting_policy == MX_WEIGHT_AREA ||
	    weighting_policy == MX_WEIGHT_AREA_AVG ) {
	    Q.set_area(norm2(e));
	    Q *= Q.area();
	}

	q(vert1) += Q;
	q(vert2) += Q;
}

uint QuadricEdgeCollapse::quadCount() { 
#if ARRAY_USE == ARRAY_NORMAL
	return quadrics.count();
#else
	return quadrics.size();
#endif
}

#endif //__ITERATIVE_QUADRIC_EDGE_COLLAPSE__