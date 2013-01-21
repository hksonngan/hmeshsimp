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
#define ARRAY_USE	1 // use hash as face and vertex array
#include "pcol_iterative.h"
#include "ecol_iterative_quadric.h"
#include "data_type.h"
#include "mc.h"
#include "vol_set.h"
#include "mcsimp_types.h"

using std::string;
using std::vector;

typedef Byte *pByte;

class MCSimp {
private:
	PairCollapse	*pcol;
	//float			decimateRate;
	//HVertex		MCCoordStart;	// the top-left coordinate of all cubes
	//HVertex		cubeLen;		// the length of each edge of the cube in three dimensions
	//HTriple<uint>	sliceCount;
	double			isovalue;
	VolumeSet		volSet;
	VertexIndexMap	vertexMap;

	// used for polygonization of cubes
	HVertex			vertlist[12];
	InterpOnWhich	onWhich[12];
	unsigned int	vertIndex[12];
	unsigned int	vertCount;

public:
	MCSimp(): pcol(NULL) { }
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

	bool genIsosurfaces(string filename, double isovalue, vector<TRIANGLE> &tris);
	bool genCollapse(string filename, double _isovalue, double decimateRate);
	void drawMesh();

private:
	inline void getVert(
		HVertex &vert, 
		DataType dataType, 
		pByte data,
		uint vertCoordFirstDimOffSet, 
		uint vertCoordSecondDimOffSet, 
		uint vertCoordThirdDimOffSet);
	XYZ vertexInterp(XYZ p1, XYZ p2, double valp1, double valp2, InterpOnWhich& onWhich);
	int polygonise(FLOAT4 gridIndex, GRIDCELL grid, TRIANGLE *triangles);
	inline unsigned int getVertIndex(const HVertex &v);
	void finalizeVert(const uint &index, const HVertex &v);

	inline bool rightMost(FLOAT4 &cubeIndex);
	inline bool backMost(FLOAT4 &cubeIndex);
	inline bool downMost(FLOAT4 &cubeIndex);
	inline bool rightBackMost(FLOAT4 &cubeIndex);
	inline bool rightDownMost(FLOAT4 &cubeIndex);
	inline bool backDownMost(FLOAT4 &cubeIndex);
	inline bool rightBackDownMost(FLOAT4 &cubeIndex);
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

unsigned int MCSimp::getVertIndex(const HVertex &v) {
	VertexIndexMap::iterator iter = vertexMap.find(v);
	if (iter == vertexMap.end()) {
		pcol->addVertex(vertCount, v, UNFINAL);
		vertexMap[v] = vertCount ++;
		return vertCount - 1;
	} else {
		return iter->second;
	}
}

bool MCSimp::rightMost(FLOAT4 &cubeIndex) {
	return cubeIndex.s[0] == volSet.volumeSize.s[0] - 2;
}

bool MCSimp::backMost(FLOAT4 &cubeIndex) {
	return cubeIndex.s[1] == volSet.volumeSize.s[1] - 2;
}

bool MCSimp::downMost(FLOAT4 &cubeIndex) {
	return cubeIndex.s[2] == volSet.volumeSize.s[2] - 2;
}

bool MCSimp::rightBackMost(FLOAT4 &cubeIndex) {
	return rightMost(cubeIndex) && backMost(cubeIndex);
}

bool MCSimp::rightDownMost(FLOAT4 &cubeIndex) {
	return rightMost(cubeIndex) && downMost(cubeIndex);
}

bool MCSimp::backDownMost(FLOAT4 &cubeIndex) {
	return backMost(cubeIndex) && downMost(cubeIndex);
}

bool MCSimp::rightBackDownMost(FLOAT4 &cubeIndex) {
	return rightMost(cubeIndex) && backMost(cubeIndex) && downMost(cubeIndex);
}

#endif