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
#include "vol_set.h"
#include "mcsimp_types.h"

using std::string;
using std::vector;
using std::ofstream;

typedef Byte *pByte;

// A Class that contains on-the-fly simplification
// And Marching Cubes generation
class MCSimp {
private:
	PairCollapse*	_m_pcol;
	double			_m_isovalue;
	VolumeSet		_m_vol_set;
	VertexIndexMap	_vertex_map;

	// used for polygonization of cubes
	HVertex			_m_vert_list[12];
	InterpOnWhich	_m_on_which[12];
	unsigned int	_m_vert_index[12];
	
	// used for decimation/oocgen
	unsigned int	_m_gen_vert_count;
	unsigned int	_m_gen_face_count;
	unsigned int	_m_new_face_count;
	const double	_m_init_decimate_rate;

	//vector<HVertex>	_m_verts;
	//vector<HFace>	faces;

	// for oocgen
	void (MCSimp::*_m_final_vert_hook)(const uint &, const HVertex &);
	ofstream		_m_vert_fout;
	// vertices that are finalized but haven't been output
	IndexVertexMap	_m_final_unput_verts;
	// the index of the vertex that should be output next
	unsigned int	_m_next_out_vert_index;

	string			_m_info;

public:
	MCSimp(double _initDecimateRate = 0.5);
	~MCSimp();

	VolumeSet* getVolSet() { return &_m_vol_set; }
	unsigned int getGenFaceCount() { return _m_gen_face_count; }
	unsigned int getGenVertCount() { return _m_gen_vert_count; }

	bool genIsosurfaces(string filename, double _isovalue, 
        int *sampleStride, vector<float> &tris, VolumeSet *paraVolSet = NULL);
	bool genIsosurfacesMT(string filename, double _isovalue, 
		int *sampleStride, vector<float> &tris, VolumeSet *paraVolSet = NULL);
	bool genCollapse(
		string filename, double _isovalue, double decimateRate, 
		int *sampleStride, unsigned int maxNewTri, unsigned int &nvert, 
        unsigned int &nface, VolumeSet *paraVolSet = NULL);
	bool oocIndexedGen(string input_file, string output_file, double _isovalue);
	void toIndexedMesh(HVertex *vertArr, HFace *faceArr);
	void toIndexedMesh(vector<float>& vertArr, vector<int>& faceArr);

	const std::string& info() { return _m_info; }
	void addInfo(const string &str) { _m_info += str; }
	void clearInfo() { _m_info = ""; }

private:
	XYZ vertexInterp(XYZ p1, XYZ p2, double valp1, double valp2, InterpOnWhich& onWhich);
	int polygonise(const UINT4& gridIndex, const GRIDCELL& grid, HFace *face);
	inline unsigned int getVertIndex(const HVertex &v);

	void genColFinalVertHook(const uint &index, const HVertex &v);
	void oocGenFinalVertHook(const uint &index, const HVertex &v);

	inline bool rightMost(const UINT4 &cubeIndex);
	inline bool backMost(const UINT4 &cubeIndex);
	inline bool downMost(const UINT4 &cubeIndex);
	inline bool rightBackMost(const UINT4 &cubeIndex);
	inline bool rightDownMost(const UINT4 &cubeIndex);
	inline bool backDownMost(const UINT4 &cubeIndex);
	inline bool rightBackDownMost(const UINT4 &cubeIndex);
};

unsigned int MCSimp::getVertIndex(const HVertex &v) {
	VertexIndexMap::iterator iter = _vertex_map.find(v);
	if (iter == _vertex_map.end()) {
		CollapsableVertex *pcv;
		_m_pcol->addVertex(_m_gen_vert_count, v, pcv);
		pcv->unfinal();
		_vertex_map[v] = _m_gen_vert_count ++;
		return _m_gen_vert_count - 1;
	} else {
		return iter->second;
	}
}

bool MCSimp::rightMost(const UINT4 &cubeIndex) {
	return _m_vol_set.rightMost(cubeIndex);
}

bool MCSimp::backMost(const UINT4 &cubeIndex) {
	return _m_vol_set.backMost(cubeIndex);
}

bool MCSimp::downMost(const UINT4 &cubeIndex) {
	return _m_vol_set.downMost(cubeIndex);
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