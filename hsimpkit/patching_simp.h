/*
 *  Divide the mesh and simplify
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __H_PATCHING_SIMP__
#define __H_PATCHING_SIMP__

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <boost/unordered_map.hpp>

#include "lru_cache.h"
#include "ply_stream.h"

#include "mesh_patch.h"
#include "grid_patch.h"
#include "hash_def.h"
#include "h_dynamarray.h"

using std::ofstream;
using std::ostringstream;
using std::streampos;
using std::string;
using std::endl;

using boost::unordered::unordered_map;


/*
 *  The dividing algorithm takes two phase:
 *  1. retrieve the bounding box and create the
 *     vertex binary file (if the input file is
 *     text file) or convert add vertices coordinates
 *     to the face record.
 *  2. partition the mesh to patches
 */

/* ========================== & DEFINITION & ======================= */

/* map between grid index and the patch object */

typedef HTriple<uint> HPatchIndex;
typedef 
unordered_map<
	HPatchIndex, HGridPatch*, 
	HTripleHash, HTripleSequencedEqual> 
HIndexPatchMap;

/* out-of-core mesh divide base on the uniform grid */
class PatchingSimp {

	/////////////////////////////////////
	// CONSTATNTS

private:
	static const uint INIT_INFO_BUF_SIZE = 4000;
	static const float MAX_OCCUPANCY;
	static const uint MAX_BUCKET_COUNT = 1000;
	static const uint MAX_CACHE_BUCKETS = 30000;
	static const uint MAX_CACHE_SIZE = 100000;

public:
	PatchingSimp() {
		vertbin_name = NULL;
		tmp_base = NULL;
	}
	~PatchingSimp() {
		if (vertbin_name)
			delete[] vertbin_name;
	}

	/////////////////////////////////////
	// PARTITION

	/* 
	 * the first pass
	 * retrieve the bounding box and create the vertex binary file
	 */
	bool readPlyFirst(char* _ply_name);
	/* 
	 * the second pass
	 * partition the mesh
	 * X Y Z: x y z axis division count 
	 */
	bool readPlySecond(uint _X, uint _Y, uint _Z);

	/////////////////////////////////////
	// SIMPLIFY
	bool simplfiyPatchesToPly(uint target_vert);
	template<class VOutType, class FOutType, class IdMapStreamType>
	bool mergeSimp(uint target_vert, VOutType &vout, FOutType &fout, 
					IdMapStreamType &bound_id_map);
	bool mergeSimpPly(uint target_vert, bool binary = true);

	/////////////////////////////////////
	// OUTPUT

	bool patchesToPly();
	
	/////////////////////////////////////
	// INFO

	const char* info() { return INFO.c_str(); }
	string& infoString() { return INFO; }
	inline void clearInfo();

	/* set the temporary file directory */
	bool tmpBase(char *s);

private:
	
	/////////////////////////////////////
	// INFO

	inline void info(ostringstream &oss);
	inline void addInfo(const char *s);
	inline void addInfo(ostringstream &oss);

	/////////////////////////////////////
	// PARTITION

	inline bool addVertexFirst(const int &i, const HVertex &v);
	inline void partitionInit();
	inline bool partitionEnd();
	inline void getSlice();
	inline void getGridIndex(const HVertex &v, HTriple<uint> &i);
	inline HGridPatch* getPatch(const HPatchIndex &pi);
	inline bool addFaceToPatch(const HTriple<uint> &face, const HVertex v1, const HVertex v2, const HVertex v3);

private:

	/////////////////////////////////////
	// PARTITION

	/* 
	 * a hash map, key is the grid coordinate, 
	 * value is the patch object 
	 */
	HIndexPatchMap 
				indexPatchMap;
	HDynamArray<HTriple<uint>> 
				patchIndices;
	HIBTriangles 
				ibt;

	/* num of vertices & faces */
	uint		vert_count;
	uint		face_count;

	/* bound box */
	float		max_x, min_x; 
	float		max_y, min_y; 
	float		max_z, min_z;

	uint		x_div, y_div, z_div;
	float		x_slice, y_slice, z_slice;

	/* file name opened */
	char		*file_name;

	/* vertex cache file */
	char		*vertbin_name;
	LRUCache<LRUVertex>	
				vert_bin;

	/* vertices count after decimation */
	uint		simp_verts;
	uint		simp_faces;

	/* the temporary file base directory */
	char		*tmp_base;
	LRUVertex	tmpv;

	/* return information */
	string		INFO;
};


/* ========================== & IMPLEMENTATION & ======================= */

void PatchingSimp::info(ostringstream &oss) {

	clearInfo();
	addInfo(oss);
}

void PatchingSimp::addInfo(const char *s) {

	INFO += s;
}

void PatchingSimp::addInfo(ostringstream &oss) {

	INFO += oss.str();
}

void PatchingSimp::clearInfo() {

	INFO.clear();
}

bool PatchingSimp::addVertexFirst(const int &i, const HVertex &v) {
	
	if (i == 0) {
		max_x = min_x = v.x;
		max_y = min_y = v.y;
		max_z = min_z = v.z;
	}
	else {
		if (max_x < v.x)
			max_x = v.x;
		else if (min_x > v.x)
			min_x = v.x;

		if (max_y < v.y)
			max_y = v.y;
		else if (min_y > v.y)
			min_y = v.y;

		if (max_z < v.z)
			max_z = v.z;
		else if (min_z > v.z)
			min_z = v.z;
	}

	/* write to vertex binary file */
	tmpv.v = v;
	vert_bin.writeVal(tmpv);

	return true;
}

void PatchingSimp::getSlice() {

	float _max_x = max_x;
	float _min_x = min_x;
	float _max_y = max_y;
	float _min_y = min_y;
	float _max_z = max_z;
	float _min_z = min_z;

	float half_range_x = (_max_x - _min_x) / 2 * 1.025;
	float half_range_y = (_max_y - _min_y) / 2 * 1.025;
	float half_range_z = (_max_z - _min_z) / 2 * 1.025;

	max_x = (_max_x + _min_x) / 2 + half_range_x;
	min_x = (_max_x + _min_x) / 2 - half_range_x;
	max_y = (_max_y + _min_y) / 2 + half_range_y;
	min_y = (_max_y + _min_y) / 2 - half_range_y;
	max_z = (_max_z + _min_z) / 2 + half_range_z;
	min_z = (_max_z + _min_z) / 2 - half_range_z;

	x_slice = (max_x - min_x) / x_div;
	y_slice = (max_y - min_y) / y_div;
	z_slice = (max_z - min_z) / z_div;
}

void PatchingSimp::partitionInit() {

	getSlice();

	/* index patch hash */
	uint hash_size = x_div * y_div * z_div * MAX_OCCUPANCY;
	if (hash_size == 0)
		hash_size = 1;
	else if (hash_size > MAX_BUCKET_COUNT)
		hash_size = MAX_BUCKET_COUNT;
	indexPatchMap.rehash(hash_size);

	/* vertex binary file */
	uint cache_buckets = vert_count * 0.3 * 0.3;
	uint cache_size = vert_count * 0.3;
	vert_bin.initCache(cache_buckets, cache_size);

	ibt.openIBTFileForWrite(tmp_base);
}

bool PatchingSimp::partitionEnd() {

	HIndexPatchMap::iterator iter;
	ostringstream oss;
	int i;

	if (ibt.closeIBTFileForWrite() == false) {
		oss.clear();
		oss << "#ERROR: close ibt file for write failed" << endl;
		info(oss);
		return false;
	}

	oss << "\t_______________________________________________" << endl
		<< "\tsecond pass complete" << endl
		<< "\tgrid size: " << x_div << "x" << y_div << "x" << z_div << endl
		<< vert_bin.info("\t")
		<< "\tibtriangles count: " << ibt.faceCount() << endl
		<< "\tpatch count: " << indexPatchMap.size() << endl
		<< "\tpatches info: " << endl
		<< "\t\tid\tindex\tverts\tibverts\tebverts\tfaces" << endl;

	vert_bin.clearCache();

	patchIndices.resize(indexPatchMap.size());
	for (iter = indexPatchMap.begin(), i = 0; iter != indexPatchMap.end(); iter ++, i ++) {

		if (iter->second->closeForWrite() == false) {
			oss.clear();
			oss << "#ERROR: close patch " << iter->first.i << "_" << iter->first.j << "_" << iter->first.k << " files for write failed" << endl;
			info(oss);
			return false;
		}

		patchIndices.push_back(iter->first);

		oss << "\t\t" << i << "\t" << iter->first.i << "_" << iter->first.j << "_" << iter->first.k 
			<< "\t" << iter->second->verts() << "\t" << iter->second->interiors()
			<< "\t" << iter->second->exteriors() << "\t" << iter->second->faces() << endl;

		delete iter->second;
	}

	info(oss);

	return true;
}

void PatchingSimp::getGridIndex(const HVertex &v, HTriple<uint> &i) {

	i.i = (int)((v.x - min_x) / x_slice);
	if (i.i >= x_div) {
		i.i = x_div - 1;
	}

	i.j = (int)((v.y - min_y) / y_slice);
	if (i.j >= y_div) {
		i.j = y_div - 1;
	}

	i.k = (int)((v.z - min_z) / z_slice);
	if (i.k >= z_div) {
		i.k = z_div - 1;
	}
}

HGridPatch* PatchingSimp::getPatch(const HPatchIndex &pi) {

	HIndexPatchMap::iterator iter;
	iter = indexPatchMap.find(pi);

	if (iter != indexPatchMap.end()) 
		return iter->second;

	HGridPatch* pPatch = new HGridPatch();
	indexPatchMap.insert(HIndexPatchMap::value_type(pi, pPatch));

	pPatch->openForWrite(tmp_base, pi);
	return pPatch;
}

bool PatchingSimp::addFaceToPatch(const HTriple<uint> &face, const HVertex v1, const HVertex v2, const HVertex v3) {

	HTriple<uint> v1pindex, v2pindex, v3pindex;
	HGridPatch *pPatch1, *pPatch2, *pPatch3;

	getGridIndex(v1, v1pindex);
	getGridIndex(v2, v2pindex);
	getGridIndex(v3, v3pindex);

	if (v1pindex == v2pindex && v1pindex == v3pindex ) {

		pPatch1 = getPatch(v1pindex);
		pPatch1->addFace(face);
	}
	else if (v1pindex == v2pindex && v1pindex != v3pindex) {

		ibt.addIBTriangle(face);

		pPatch1 = getPatch(v1pindex);
		pPatch2 = getPatch(v3pindex);

		pPatch1->addFace(face);
		pPatch2->addFace(face);

		pPatch1->addInteriorBound(face.i);
		pPatch1->addInteriorBound(face.j);
		pPatch1->addExteriorBound(face.k, v3);

		pPatch2->addInteriorBound(face.k);
		pPatch2->addExteriorBound(face.i, v1);
		pPatch2->addExteriorBound(face.j, v2);
	}
	else if (v1pindex == v3pindex && v1pindex != v2pindex) {

		ibt.addIBTriangle(face);

		pPatch1 = getPatch(v1pindex);
		pPatch2 = getPatch(v2pindex);

		pPatch1->addFace(face);
		pPatch2->addFace(face);

		pPatch1->addInteriorBound(face.i);
		pPatch1->addInteriorBound(face.k);
		pPatch1->addExteriorBound(face.j, v2);

		pPatch2->addInteriorBound(face.j);
		pPatch2->addExteriorBound(face.i, v1);
		pPatch2->addExteriorBound(face.k, v3);
	}
	else if (v2pindex == v3pindex && v1pindex != v2pindex) {

		ibt.addIBTriangle(face);

		pPatch1 = getPatch(v2pindex);
		pPatch2 = getPatch(v1pindex);

		pPatch1->addFace(face);
		pPatch2->addFace(face);

		pPatch1->addInteriorBound(face.j);
		pPatch1->addInteriorBound(face.k);
		pPatch1->addExteriorBound(face.i, v1);

		pPatch2->addInteriorBound(face.i);
		pPatch2->addExteriorBound(face.j, v2);
		pPatch2->addExteriorBound(face.k, v3);
	}
	else /*(v1pindex != v2pindex && v1pindex != v3pindex)*/ {

		ibt.addIBTriangle(face);

		pPatch1 = getPatch(v1pindex);
		pPatch2 = getPatch(v2pindex);
		pPatch3 = getPatch(v3pindex);

		pPatch1->addFace(face);
		pPatch2->addFace(face);
		pPatch3->addFace(face);

		pPatch1->addInteriorBound(face.i);
		pPatch1->addExteriorBound(face.j, v2);
		pPatch1->addExteriorBound(face.k, v3);

		pPatch2->addInteriorBound(face.j);
		pPatch2->addExteriorBound(face.i, v1);
		pPatch2->addExteriorBound(face.k, v3);

		pPatch3->addInteriorBound(face.k);
		pPatch3->addExteriorBound(face.i, v1);
		pPatch3->addExteriorBound(face.j, v2);
	}

	return true;
}

#endif //__H_PATCHING_SIMP__