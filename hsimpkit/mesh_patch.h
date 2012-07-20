/*
 *  The patch class in a patching simplification
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __H_MESH_PATCH__
#define __H_MESH_PATCH__

#include <iostream>
#include <fstream>
#include <sstream>
#include <list>

#include <boost/unordered_map.hpp>

#include "util_common.h"
#include "trivial.h"
#include "pcol_iterative.h"
#include "os_dependent.h"

using std::ostream;
using std::ofstream;
using std::ifstream;
using std::fstream;
using std::ostringstream;
using std::list;

using boost::unordered::unordered_map;


/*
 *  interior boundary triangle: the triangle which has some vertices belonging to 
 *	  different patches
 *
 *  interior boundary vertex: the vertex belonging to the patch while adjacent to 
 *    interior boundary triangle
 *
 *  exterior boundary vertex: the vertex not belonging to the patch while adjacent
 *    to some interior boundary triangles which has some vertices belonging to the 
 *    patch
 */

/* ================================ & DEFINITION & ============================= */

#define VERT_ITEM_SIZE sizeof(float)

enum TargetOption { target_face, target_vert };

/* interior boundary triangles */
class HIBTriangles {
public:
	HIBTriangles() { face_count = 0; }

	bool openIBTFileForWrite(const char* dir_path);
	inline bool addIBTriangle(const HTripleIndex<uint> &f);
	bool closeIBTFileForWrite();

	bool openIBTFileForRead(const char* dir_path);
	inline bool nextIBTriangle(HTripleIndex<uint> &f);
	bool closeIBTFileForRead();

	uint faceCount() const { return face_count; }

public:
	ofstream ibt_out;
	ifstream ibt_in;
	uint face_count;
};

/* helper data object */
class HIdVertex {
public:
	bool operator < (const HIdVertex &v) const {
		return id < v.id;
	}

	bool operator == (const HIdVertex &v) const {
		return id == v.id;
	}

	void write(ostream &out) {

		WRITE_UINT(out, id);
		WRITE_BLOCK(out, v.x, VERT_ITEM_SIZE);
		WRITE_BLOCK(out, v.y, VERT_ITEM_SIZE);
		WRITE_BLOCK(out, v.z, VERT_ITEM_SIZE);
	}

public:
	uint	id;
	HVertex	v;
};

/* a generic patch class */
class HMeshPatch {

public:

	HMeshPatch() { vert_count = 0; face_count = 0; interior_count = 0; exterior_count = 0; }

	/////////////////////////////////
	// WRITE

	bool openForWrite(const char* vert_name, const char* face_name);
	bool closeForWrite();
	inline bool addInteriorVertex(const uint &orig_id, const HVertex &v);
	inline void addInteriorBound(const uint &orig_id);
	inline void addExteriorBound(const uint &orig_id, const HVertex &v);
	inline bool addFace(const HTripleIndex<uint> &f);

	/////////////////////////////////
	// READ

	bool openForRead(const char* vert_name, const char* face_name);
	bool closeForRead();
	inline bool nextInteriorVertex(uint &orig_id, HVertex &v);
	inline bool nextInteriorBound(uint &orig_id);
	inline bool nextExteriorBound(uint &orig_id, HVertex &v);
	inline bool nextFace(HTripleIndex<uint> &f);

	/////////////////////////////////
	// SIMPLIFY

	bool readPatch(char *vert_patch, char *face_patch, PairCollapse *pcol);
	bool toPly(PairCollapse *pcol, const char* ply_name);
	inline bool writeVert(ostream &out);
	template<class OutType> bool pairCollapse(
			char *vert_name, char *face_name, uint vert_start_id,
			uint target, ostream &vout, OutType fout);

	/////////////////////////////////
	// ACCESSORS

	uint interiors() const { return interior_count; }
	uint exteriors() const { return exterior_count; }
	uint verts() const { return vert_count; }
	uint faces() const { return face_count; }

public:
	/* interior vertices count */
	uint vert_count;
	/* interior boundary vertices count */
	uint interior_count;
	/* exterior boundary vertices count */
	uint exterior_count;
	uint face_count;

	list<uint> interior_bound;
	list<HIdVertex> exterior_bound;

	/* map between external id and internal id */
	unordered_map<uint, uint> id_map;

private:
	ofstream vert_out;
	ofstream face_out;
	ifstream vert_in;
	ifstream face_in;
};


/* ================================ & IMPLEMENTATION & ============================= */

/* -- HMeshPatch -- */

bool HMeshPatch::addInteriorVertex(const uint &orig_id, const HVertex &v) {

	WRITE_UINT(vert_out, orig_id);
	WRITE_BLOCK(vert_out, v.x, VERT_ITEM_SIZE);
	WRITE_BLOCK(vert_out, v.y, VERT_ITEM_SIZE);
	WRITE_BLOCK(vert_out, v.z, VERT_ITEM_SIZE);
	vert_count ++;
#ifndef WRITE_PATCH_BINARY
	vert_out << endl;
#endif

	if (vert_out.good())
		return true;

	cerr << "#ERROR: writing patch vertex " << orig_id << " failed" << endl;
	return false;
}

void HMeshPatch::addInteriorBound(const uint &orig_id) {

	interior_bound.push_back(orig_id);
}

void HMeshPatch::addExteriorBound(const uint &orig_id, const HVertex &v) {

	HIdVertex idv;
	idv.id = orig_id;
	idv.v = v;
	exterior_bound.push_back(idv);
}

bool HMeshPatch::addFace(const HTripleIndex<uint> &f) {

	WRITE_UINT(face_out, f.i);
	WRITE_UINT(face_out, f.j);
	WRITE_UINT(face_out, f.k);
	face_count ++;
#ifndef WRITE_PATCH_BINARY
	face_out << endl;
#endif

	if (face_out.good())
		return true;
	cerr << "#ERROR: writing patch face <" << f.i << ", " << f.j << ", " << f.k << "> failed" << endl;
	return false;
}

bool HMeshPatch::nextInteriorVertex(uint &orig_id, HVertex &v) {

	READ_UINT(vert_in, orig_id);
	READ_BLOCK(vert_in, v.x, VERT_ITEM_SIZE);
	READ_BLOCK(vert_in, v.y, VERT_ITEM_SIZE);
	READ_BLOCK(vert_in, v.z, VERT_ITEM_SIZE);

	if (vert_in.good())
		return true;

	cerr << "#ERROR: reading patch vertex failed" << endl;
	return false;
}

bool HMeshPatch::nextInteriorBound(uint &orig_id) {

	READ_UINT(vert_in, orig_id);

	if (vert_in.good())
		return true;

	cerr << "#ERROR: reading interior boundary vertex failed" << endl;
	return false;
}

bool HMeshPatch::nextExteriorBound(uint &orig_id, HVertex &v) {

	READ_UINT(vert_in, orig_id);
	READ_BLOCK(vert_in, v.x, VERT_ITEM_SIZE);
	READ_BLOCK(vert_in, v.y, VERT_ITEM_SIZE);
	READ_BLOCK(vert_in, v.z, VERT_ITEM_SIZE);

	if (vert_in.good())
		return true;

	cerr << "#ERROR: reading exterior boundary vertex failed" << endl;
	return false;
}

bool HMeshPatch::nextFace(HTripleIndex<uint> &f) {

	READ_UINT(face_in, f.i);
	READ_UINT(face_in, f.j);
	READ_UINT(face_in, f.k);

	if (face_in.good())
		return true;
	cerr << "#ERROR: reading patch face failed" << endl;
	return false;
}


/* -- HIBTriangles -- */

bool HIBTriangles::addIBTriangle(const HTripleIndex<uint> &f) {
	
	WRITE_UINT(ibt_out, f.i);
	WRITE_UINT(ibt_out, f.j);
	WRITE_UINT(ibt_out, f.k);
	face_count ++;
#ifndef WRITE_PATCH_BINARY
	ibt_out << endl;
#endif

	if (ibt_out.good())
		return true;
	cerr << "#ERROR: writing interior boundary triangle failed" << endl;
	return false;
}

bool HIBTriangles::nextIBTriangle(HTripleIndex<uint> &f) {

	READ_BLOCK(ibt_in, f.i, VERT_ITEM_SIZE);
	READ_BLOCK(ibt_in, f.i, VERT_ITEM_SIZE);
	READ_BLOCK(ibt_in, f.i, VERT_ITEM_SIZE);

	if (ibt_out.good())
		return true;
	cerr << "#ERROR: writing interior boundary triangle failed" << endl;
	return false;
}

#endif //__H_MESH_PATCH__