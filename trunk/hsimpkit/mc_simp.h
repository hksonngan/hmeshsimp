/*
 *  Incrementally simplify the mesh generated 
 *  by a Marching-cube like algorithm
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef _MC_SIMP_HEADER_
#define _MC_SIMP_HEADER_

#include <string>
#include <vector>
#include "pcol_iterative.h"
#include "ecol_iterative_quadric.h"
#include "data_type.h"
#include "mc.h"
#include "raw_set.h"
#include "mcsimp_types.h"

using std::string;
using std::vector;

typedef Byte *pByte;

class MCSimp {
private:
	//PairCollapse	*pcol;
	//float			decimateRate;
	//HVertex			MCCoordStart;	// the top-left coordinate of all cubes
	//HVertex			cubeLen;		// the length of each edge of the cube in three dimensions
	//HTriple<uint>	sliceCount;
	double			isolevel;
	RawSet			rawSet;
	VertexIndexMap	vertexMap;

public:
	MCSimp() { }
	//MCSimp(
	//	float _decimateRate, 
	//	HTriple<uint> _sliceCount,
	//	HVertex _cubeLen, 
	//	HVertex *_pMCCoordStart = NULL);
	~MCSimp();

	bool addTriangles(
		Byte *data, 
		uint nTri, 
		uint triSize,  
		uint vertCoordFirstDimOffSet, 
		uint vertCoordSecondDimOffSet, 
		uint vertCoordThirdDimOffSet,
		DATA_TYPE coordDataType = DFLOAT);

	RawSet* getRawSet() { return &rawSet; }
	bool genIsosurfaces(string filename, double isovalue, vector<TRIANGLE> &tris);
	bool genDecimate(string filename, double isovalue, double decimateRate);
	void drawMesh();

private:
	inline void getVert(
		HVertex &vert, 
		DataType dataType, 
		pByte data,
		uint vertCoordFirstDimOffSet, 
		uint vertCoordSecondDimOffSet, 
		uint vertCoordThirdDimOffSet);
	XYZ vertexInterp(double isolevel, XYZ p1, XYZ p2, double valp1, double valp2);
	int polygonise(HTriple<uint> cubeIndex, GRIDCELL grid, TRIANGLE *triangles);
};

void MCSimp::getVert(
	HVertex &vert, 
	DataType dataType, 
	pByte data,
	uint vertCoordFirstDimOffSet, 
	uint vertCoordSecondDimOffSet, 
	uint vertCoordThirdDimOffSet) {
	vert.x = dataType.getValue<float>(data + vertCoordFirstDimOffSet);
	vert.y = dataType.getValue<float>(data + vertCoordSecondDimOffSet);
	vert.z = dataType.getValue<float>(data + vertCoordThirdDimOffSet);
}

#endif