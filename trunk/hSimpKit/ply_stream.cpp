#include "ply_stream.h"
#include "ply/ply.h"
#include <iostream>

using std::cerr;
using std::endl;

PlyProperty vert_props[] = { /* list of property information for a vertex */
	{"x", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,x), 0, 0, 0, 0},
	{"y", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,y), 0, 0, 0, 0},
	{"z", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,z), 0, 0, 0, 0},
};

PlyProperty face_props[] = { /* list of property information for a vertex */
	{"vertex_indices", PLY_INT, PLY_INT, offsetof(Face,verts),
	1, PLY_UCHAR, PLY_UCHAR, offsetof(Face,nverts)},
};


PlyStream::PlyStream()
{
	ply = NULL;
	vCount = 0;
	fCount = 0;
}

bool PlyStream::openForRead(char *filename)
{
	int file_type;
	int nelems;
	char **elist;
	float version;
	int nprops;
	int num_elems;
	PlyProperty **plist = NULL;
	char *elem_name;

	int i;

	/* a space for PlyFile is malloced in the function
	   it must be freed after using, as well as elist - houtao */
	ply = ply_open_for_reading(filename, &nelems, &elist, &file_type, &version);
	if(ply == NULL) {
		cerr << "#error: ply file open failed" << endl;
		return false;
	}

	for (i = 0; i < nelems; i++) {

		/* get the description of the first element */
		elem_name = elist[i];
		/* the prop_list as well as the property in it and the char* name in the property is malloced - houtao */
		plist = ply_get_element_description (ply, elem_name, &num_elems, &nprops);

		if (equal_strings ("vertex", elem_name)) {
			ply_get_property (ply, elem_name, &vert_props[0]);
			ply_get_property (ply, elem_name, &vert_props[1]);
			ply_get_property (ply, elem_name, &vert_props[2]);
			vCount = num_elems;
		}
		else if (equal_strings ("face", elem_name)) {
			fCount = num_elems;
			ply_get_property (ply, elem_name, &face_props[0]);
		}

		/* free plist and its memory space in case of memory leaking */
		if (plist) {
			for (int i = 0; i < nprops; i ++) {
				if (plist[i]) {
					if (plist[i]->name) {
						free(plist[i]->name); 
					}
					free(plist[i]);
				}
			}
			free(plist);
		}
	}

	readVCount = 0;
	readFCount = 0;

	return true;
}

bool PlyStream::close()
{
	ply_close(ply);

	return true;
}

inline bool PlyStream::nextVertex(HVertex &v)
{
	if (readVCount >= vCount) {
		return false;
	}
	
	vertex.other_props = NULL;
	ply_get_element (ply, (void *) &vertex);

	v.Set(vertex.x, vertex.y, vertex.z);
	readVCount ++;

	if (vertex.other_props) {
		free(vertex.other_props);
	}

	return true;
}

static void freePointersInFace(Face *face) {
	if (face->verts) {
		free(face->verts);
	}
	if (face->other_props) {
		free(face->other_props);
	}
}

inline bool PlyStream::nextFace(HTripleIndex &f)
{
	if (readFCount >= fCount) {
		return false;
	}

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
