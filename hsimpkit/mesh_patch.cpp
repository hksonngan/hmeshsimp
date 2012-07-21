#include <algorithm>
#include "mesh_patch.h"
#include "os_dependent.h"
#include "h_dynamarray.h"
#include "ecol_iterative_quadric.h"

using std::cerr;
using std::endl;
using std::streampos;


#define __FILE_NAME_BUF_SIZE 400

/* -- HMeshPatch -- */

bool HMeshPatch::openForWrite(const char* vert_name, const char* face_name) {

	uint n = 0;

	vert_count = 0;
	face_count = 0;
	interior_count = 0;
	exterior_count = 0;
	interior_bound.clear();
	exterior_bound.clear();

	vert_out.open(vert_name, fstream::binary | fstream::out);
	if (!vert_out.good()) {
		cerr << "#ERROR: " << vert_name << " vertex file open for write failed" << endl;
		return false;
	}
#ifdef WRITE_PATCH_BINARY
	// count of vertices
	WRITE_UINT(vert_out, n);
	// count of interior boundary vertices
	WRITE_UINT(vert_out, n);
	// count of exterior boundary vertices
	WRITE_UINT(vert_out, n);
#endif

	face_out.open(face_name, fstream::binary | fstream::out);
	if (!face_out.good()) {
		cerr << "#ERROR: " << face_name << " face file open for write failed" << endl;
		return false;
	}
#ifdef WRITE_PATCH_BINARY
	// count of faces
	WRITE_UINT(face_out, n);
#endif

	return true;
}

bool HMeshPatch::closeForWrite() {

	HDynamArray<uint> interior;
	HDynamArray<HIdVertex> exterior;
	list<uint>::iterator interior_iter;
	list<HIdVertex>::iterator exterior_iter;
	int i, count;
	HIdVertex idv;
	uint n = 0;

	/* write interior and exterior boundary vertices */

	interior.clear();
	interior.resize(interior_bound.size());
	for (interior_iter = interior_bound.begin(); interior_iter != interior_bound.end(); interior_iter ++) 
		interior.push_back(*interior_iter);
	interior_bound.clear();
	std::sort(interior.pointer(0), interior.pointer(0) + interior.count());

#ifndef WRITE_PATCH_BINARY
	vert_out << endl;
#endif

	for (i = 0, interior_count = 0; i < interior.count(); interior_count ++) {

		n = interior[i];
		i ++;

		// write the vertex id
		WRITE_UINT(vert_out, n);
#ifndef WRITE_PATCH_BINARY
		vert_out << endl;
#endif
		if (!vert_out.good()) {
			cerr << "#ERROR: write interior boundary vertex " << n << " failed" << endl;
			return false;
		}
		
		// in case of duplication
		for (; i < interior.count() && interior[i] == n; i ++);
	}
#ifndef WRITE_PATCH_BINARY
	vert_out << endl;
#endif
	interior.freeSpace();
	
	exterior.clear();
	exterior.resize(exterior_bound.size());
	for (exterior_iter = exterior_bound.begin(); exterior_iter != exterior_bound.end(); exterior_iter ++) 
		exterior.push_back(*exterior_iter);
	std::sort(exterior.pointer(0), exterior.pointer(0) + exterior.count());
	exterior_bound.clear();

	for (i = 0, exterior_count = 0; i < exterior.count(); exterior_count ++) {

		idv = exterior[i];
		i ++;

		// write the vertex
		idv.write(vert_out);
#ifndef WRITE_PATCH_BINARY
		vert_out << endl;
#endif
		if (!vert_out.good()) {
			cerr << "#ERROR: write exterior boundary vertex " << idv.id << " failed" << endl;
			return false;
		}

		// in case of duplication
		for (; i < exterior.count() && exterior[i] == idv; i ++);
	}

#ifndef WRITE_PATCH_BINARY
	vert_out << endl;
#endif

#ifdef WRITE_PATCH_BINARY
	// write the counts
	vert_out.seekp(0);
	WRITE_UINT(vert_out, vert_count);
	WRITE_UINT(vert_out, interior_count);
	WRITE_UINT(vert_out, exterior_count);
#endif

	vert_out.close();
	if (!vert_out.good()) {
		cerr << "#ERROR: close vertex file failed" << endl;
		return false;
	}

	/* write face count */

#ifdef WRITE_PATCH_BINARY
	face_out.seekp(0);
	WRITE_UINT(face_out, face_count);
#endif

	face_out.close();
	if (!face_out.good()) {
		cerr << "#ERROR: close face file failed" << endl;
		return false;
	}

	return true;
}

bool HMeshPatch::openForRead(const char* vert_name, const char* face_name) {

	vert_in.open(vert_name, fstream::binary | fstream::in);
	if (!vert_in.good()) {
		cerr << "#ERROR: " << vert_name << " vertex file open for read failed" << endl;
		return false;
	}
#ifdef WRITE_PATCH_BINARY
	// count of vertices
	READ_UINT(vert_in, vert_count);
	// count of interior boundary vertices
	READ_UINT(vert_in, interior_count);
	// count of exterior boundary vertices
	READ_UINT(vert_in, exterior_count);
#else
	vert_count = 0;
	interior_count = 0;
	exterior_count = 0;
#endif
	interior_bound.clear();
	exterior_bound.clear();
	id_map.clear();

	face_in.open(face_name, fstream::binary | fstream::in);
	if (!face_in.good()) {
		cerr << "#ERROR: " << face_name << " face file open for read failed" << endl;
		return false;
	}
#ifdef WRITE_PATCH_BINARY
	// count of faces
	READ_UINT(face_in, face_count);
#else
	face_count = 0;
#endif

	return true;
}

bool HMeshPatch::closeForRead() {

	vert_in.close();
	face_in.close();

	return true;
}

bool HMeshPatch::readPatch(char *vert_patch, char *face_patch, PairCollapse *pcol) {

	int i;
	uint orig_id; // external id
	HVertex v;
	HTriple<uint> f, f_internal;

	if (!HMeshPatch::openForRead(vert_patch, face_patch))
		return false;
	pcol->allocVerts(vert_count + exterior_count);
	pcol->allocFaces(face_count);

	for (i = 0; i < vert_count; i ++) {
		if (!nextInteriorVertex(orig_id, v))
			return false;
		id_map[orig_id] = i;
		pcol->addVertex(v);
	}

	for (i = 0; i < interior_count; i ++) {
		if (!nextInteriorBound(orig_id))
			return false;
		(pcol->v(id_map[orig_id])).markv(INTERIOR_BOUND);
	}

	for (i = 0; i < exterior_count; i ++) {
		if (!nextExteriorBound(orig_id, v))
			return false;
		id_map[orig_id] = vert_count + i;
		pcol->addVertex(v);
		(pcol->v(vert_count + i)).markv(EXTERIOR);
	}

	for (i = 0; i < face_count; i ++) {
		if (!nextFace(f))
			return false;
		f_internal.i = id_map[f.i];
		f_internal.j = id_map[f.j];
		f_internal.k = id_map[f.k];
		pcol->addFace(f_internal);
	}

	closeForRead();

	return true;
}

bool HMeshPatch::toPly(PairCollapse *pcol, const char* ply_name) {

	ofstream fout(ply_name);
	if (fout.bad())
		return false;

	/* write head */
	fout << "ply" << endl;
	fout << "format ascii 1.0" << endl;
	fout << "comment generated by patching simp" << endl;

	fout << "element vertex " << pcol->validVerts() << endl;
	fout << "property float x" << endl;
	fout << "property float y" << endl;
	fout << "property float z" << endl;
	fout << "element face " << pcol->validFaces() << endl;
	fout << "property list uchar int vertex_indices" << endl;
	fout << "end_header" << endl;

	int i;
	uint valid_count = 0;

	for (i = 0; i < pcol->vertexCount(); i ++) { 
		CollapsableVertex &v = pcol->v(i);
		if (v.valid(i) && !v.unreferred()) {
			fout << v.x << " " << v.y << " " << v.z << endl;
			v.setOutId(valid_count);
			valid_count ++;
		}
	}

	for (i = 0; i < pcol->faceCount(); i ++) {
		CollapsableFace &f = pcol->f(i);
		if (f.valid()) {
			fout << "3 " << pcol->v(f.i).output_id << " " << pcol->v(f.j).output_id << " "
				<< pcol->v(f.k).output_id << endl;
		}
	}

	return true;
}

/* -- HIBTriangles -- */

bool HIBTriangles::openIBTFileForWrite(const char* dir_path) {

	char buf[__FILE_NAME_BUF_SIZE];
	uint n = 0;

	string str;
	if (dir_path) {
		str = dir_path;
		str += hPathSeperator();
	}
	str += "interior_boundary_triangles";
	stringToCstr(str, buf);

	ibt_out.open(buf, fstream::binary | fstream::out);
#ifdef WRITE_PATCH_BINARY
	// count of triangles
	WRITE_UINT(ibt_out, n);
#endif
	if (ibt_out.good())
		return true;
	cerr << "#ERROR: open interior boundary triangles file for write failed" << endl;
	return false;
}

bool HIBTriangles::closeIBTFileForWrite() {

#ifdef WRITE_PATCH_BINARY
	// write the count of triangles to the beginning of the file
	ibt_out.seekp(0);
	WRITE_UINT(ibt_out, face_count);
#endif

	ibt_out.close();
	if (ibt_out.good())
		return true;
	cerr << "#ERROR: close interior boundary triangles file for write failed" << endl;
	return false;
}

bool HIBTriangles::openIBTFileForRead(const char* dir_path) {

	char buf[__FILE_NAME_BUF_SIZE];

	string str;
	if (dir_path) {
		str = dir_path;
		str += hPathSeperator();
	}
	str += "interior_boundary_triangles";
	stringToCstr(str, buf);

	ibt_in.open(buf, fstream::binary | fstream::in);
#ifdef WRITE_PATCH_BINARY
	// count of triangles
	READ_UINT(ibt_in, face_count);
#else
	face_count = 0;
#endif

	if (ibt_out.good())
		return true;
	cerr << "#ERROR: open interior boundary triangles file for read failed" << endl;
	return false;
}

bool HIBTriangles::closeIBTFileForRead() {

	ibt_out.close();
	return true;
}