/*
 *  Ply read & write
 * 
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */

#ifndef __PLY_STREAM__
#define __PLY_STREAM__

#include "ply/ply.h"
#include "util_common.h"
#include <iostream>

typedef struct Vertex {
	float x,y,z;
	void *other_props;       /* other properties */
} Vertex;

typedef struct Face {
	unsigned char nverts;    /* number of vertex indices in list */
	int *verts;              /* vertex index list */
	void *other_props;       /* other properties */
} Face;

extern /*"C" {*/ PlyElement *find_element(PlyFile *plyfile, char *element); /*}*/

using std::cerr;
using std::endl;

static PlyProperty vert_props[] = { /* list of property information for a vertex */
	{"x", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,x), 0, 0, 0, 0},
	{"y", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,y), 0, 0, 0, 0},
	{"z", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,z), 0, 0, 0, 0},
};

static PlyProperty face_props[] = { /* list of property information for a vertex */
	{"vertex_indices", PLY_INT, PLY_INT, offsetof(Face,verts),
	1, PLY_UCHAR, PLY_UCHAR, offsetof(Face,nverts)},
};

class PlyStream
{
public:
	PlyStream();
	bool openForRead(const char *filename);
	bool close();
	inline bool nextVertex(HVertex &v);
	inline bool nextFace(HFace &f);
	//inline bool nextFace(HTripleIndex<uint> &f);

	uint getVertexCount() const { return vCount; }
	uint getFaceCount() const { return fCount; }

	PlyFile* plyFile() const { return ply; }

private:
	PlyFile *ply;
	uint vCount;
	uint fCount;
	uint readVCount;
	uint readFCount;
	Vertex vertex;
	Face face;
	PlyElement *vertexElem;
	PlyElement *faceElem;

	HFace _face;
};

static inline void freePointersInFace(Face *face) {
	if (face->verts) {
		free(face->verts);
	}
	if (face->other_props) {
		free(face->other_props);
	}
}

bool PlyStream::nextFace(HFace &f)
{
	if (readFCount >= fCount) {
		return false;
	}

	ply->which_elem = faceElem;

	face.nverts = NULL;
	face.other_props = NULL;
	ply_get_element (ply, (void *) &face);

	if (face.nverts != 3) {
		cerr << "#error: non-tirangle in ply file" << endl;
		return false;
	}

	f.set(face.verts[0], face.verts[1], face.verts[2]);
	readFCount ++;

	freePointersInFace(&face);

	return true;
}

//bool PlyStream::nextFace(HTripleIndex<uint> &f) {
//	if (!nextFace(_face))
//		return false;
//	f.set(_face.i, _face.j, _face.k);
//
//	return true;
//}

bool PlyStream::nextVertex(HVertex &v)
{
	if (readVCount >= vCount) {
		return false;
	}

	ply->which_elem = vertexElem;

	vertex.other_props = NULL;
	ply_get_element (ply, (void *) &vertex);

	v.Set(vertex.x, vertex.y, vertex.z);
	readVCount ++;

	if (vertex.other_props) {
		free(vertex.other_props);
	}

	return true;
}

#endif //__PLY_STREAM__