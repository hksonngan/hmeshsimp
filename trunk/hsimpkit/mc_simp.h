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
	PairCollapse*	pcol;
	double			isovalue;
	VolumeSet		volSet;
	VertexIndexMap	vertexMap;

	// used for polygonization of cubes
	HVertex			vertlist[12];
	InterpOnWhich	onWhich[12];
	unsigned int	vertIndex[12];
	
	// used for decimation
	unsigned int	genVertCount;
	unsigned int	genFaceCount;
	unsigned int	newFaceCount;
	const double	initDecimateRate;

	vector<HVertex>	verts;
	vector<HFace>	faces;

	string			INFO;

public:
	MCSimp(double _initDecimateRate = 0.5);
	~MCSimp();

	VolumeSet* getVolSet() { return &volSet; }
	unsigned int getGenFaceCount() { return genFaceCount; }
	unsigned int getGenVertCount() { return genVertCount; }

	bool genIsosurfaces(string filename, double _isovalue, vector<TRIANGLE> &tris);
	bool genCollapse(
			string filename, double _isovalue, double decimateRate, 
			unsigned int maxNewTri, unsigned int &nvert, unsigned int &nface);
	void toIndexedMesh(HVertex *vertArr, HFace *faceArr);

	string& info() { return INFO; }
	void addInfo(const string &str) { INFO += str; }
	void clearInfo() { INFO = ""; }

private:
	XYZ vertexInterp(XYZ p1, XYZ p2, double valp1, double valp2, InterpOnWhich& onWhich);
	void polygonise(const UINT4& gridIndex, const GRIDCELL& grid);
	inline unsigned int getVertIndex(const HVertex &v);
	void finalizeVert(const uint &index, const HVertex &v);

	inline bool rightMost(const UINT4 &cubeIndex);
	inline bool backMost(const UINT4 &cubeIndex);
	inline bool downMost(const UINT4 &cubeIndex);
	inline bool rightBackMost(const UINT4 &cubeIndex);
	inline bool rightDownMost(const UINT4 &cubeIndex);
	inline bool backDownMost(const UINT4 &cubeIndex);
	inline bool rightBackDownMost(const UINT4 &cubeIndex);
};

unsigned int MCSimp::getVertIndex(const HVertex &v) {
	VertexIndexMap::iterator iter = vertexMap.find(v);
	if (iter == vertexMap.end()) {
		CollapsableVertex *pcv;
		pcol->addVertex(genVertCount, v, pcv);
		pcv->unfinal();
		vertexMap[v] = genVertCount ++;
		return genVertCount - 1;
	} else {
		return iter->second;
	}
}

bool MCSimp::rightMost(const UINT4 &cubeIndex) {
	return cubeIndex.s[0] == volSet.volumeSize.s[0] - 2;
}

bool MCSimp::backMost(const UINT4 &cubeIndex) {
	return cubeIndex.s[1] == volSet.volumeSize.s[1] - 2;
}

bool MCSimp::downMost(const UINT4 &cubeIndex) {
	return cubeIndex.s[2] == volSet.volumeSize.s[2] - 2;
}

bool MCSimp::rightBackMost(const UINT4 &cubeIndex) {
	return rightMost(cubeIndex) && backMost(cubeIndex);
}

bool MCSimp::rightDownMost(const UINT4 &cubeIndex) {
	return rightMost(cubeIndex) && downMost(cubeIndex);
}

bool MCSimp::backDownMost(const UINT4 &cubeIndex) {
	return backMost(cubeIndex) && downMost(cubeIndex);
}

bool MCSimp::rightBackDownMost(const UINT4 &cubeIndex) {
	return rightMost(cubeIndex) && backMost(cubeIndex) && downMost(cubeIndex);
}

#endif