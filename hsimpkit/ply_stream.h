/*
 *  Ply read & write
 * 
 *  Author: Ht
 *  Email : waytofall916@gmail.com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 *	
 *  This file is part of hmeshsimp.
 *
 *  hmeshsimp is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  hmeshsimp is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with hmeshsimp.  If not, see <http://www.gnu.org/licenses/>.
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
	bool openForRead(char *filename);
	bool close();
	inline bool nextVertex(HVertex &v);
	inline bool nextFace(HFace &f);
	inline bool nextFace(HTripleIndex<Integer> &f);

	Integer getVertexCount() { return vCount; };
	Integer getFaceCount() { return fCount; };

private:
	PlyFile *ply;
	Integer vCount;
	Integer fCount;
	Integer readVCount;
	Integer readFCount;
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

bool PlyStream::nextFace(HTripleIndex<Integer> &f) {
	if (!nextFace(_face))
		return false;
	f.set(_face.i, _face.j, _face.k);

	return true;
}

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