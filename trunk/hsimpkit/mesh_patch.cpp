#include <algorithm>
#include "mesh_patch.h"
#include "os_dependent.h"
#include "h_dynamarray.h"

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
	// count of vertices
	WRITE_UINT(vert_out, n);
	// count of interior boundary vertices
	WRITE_UINT(vert_out, n);
	// count of exterior boundary vertices
	WRITE_UINT(vert_out, n);


	face_out.open(face_name, fstream::binary | fstream::out);
	if (!face_out.good()) {
		cerr << "#ERROR: " << face_name << " face file open for write failed" << endl;
		return false;
	}
	// count of faces
	face_out.write((char *)&n, sizeof(uint));

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

	for (i = 0, interior_count = 0; i < interior.count(); interior_count ++) {

		n = interior[i];
		i ++;

		// write the vertex id
		vert_out.write((char *)&n, sizeof(uint));
		if (!vert_out.good()) {
			cerr << "#ERROR: write interior boundary vertex " << n << " failed" << endl;
			return false;
		}
		
		// in case of duplication
		for (; i < interior.count() && interior[i] == n; i ++);
	}
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
		if (!vert_out.good()) {
			cerr << "#ERROR: write exterior boundary vertex " << idv.id << " failed" << endl;
			return false;
		}

		// in case of duplication
		for (; i < exterior.count() && exterior[i] == idv; i ++);
	}

	// write the counts
	vert_out.seekp(0);
	WRITE_UINT(vert_out, vert_count);
	WRITE_UINT(vert_out, interior_count);
	WRITE_UINT(vert_out, exterior_count);

	vert_out.close();
	if (!vert_out.good()) {
		cerr << "#ERROR: close vertex file failed" << endl;
		return false;
	}

	/* write face count */

	face_out.seekp(0);
	WRITE_UINT(face_out, face_count);

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
	// count of vertices
	READ_UINT(vert_in, vert_count);
	// count of interior boundary vertices
	READ_UINT(vert_in, interior_count);
	// count of exterior boundary vertices
	READ_UINT(vert_in, exterior_count);

	face_in.open(face_name, fstream::binary | fstream::in);
	if (!face_in.good()) {
		cerr << "#ERROR: " << face_name << " face file open for read failed" << endl;
		return false;
	}
	// count of faces
	READ_UINT(face_in, face_count);

	return true;
}

bool HMeshPatch::closeForRead() {

	vert_in.close();
	face_in.close();

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
	// count of triangles
	WRITE_UINT(ibt_out, n);
	if (ibt_out.good())
		return true;
	cerr << "#ERROR: open interior boundary triangles file for write failed" << endl;
	return false;
}

bool HIBTriangles::closeIBTFileForWrite() {

	// write the count of triangles to the beginning of the file
	ibt_out.seekp(0);
	WRITE_UINT(ibt_out, face_count);

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
	// count of triangles
	READ_UINT(ibt_in, face_count);

	if (ibt_out.good())
		return true;
	cerr << "#ERROR: open interior boundary triangles file for read failed" << endl;
	return false;
}

bool HIBTriangles::closeIBTFileForRead() {

	ibt_out.close();
	return true;
}


/* -- HGridPatch -- */

bool HGridPatch::openForWrite(const char* dir_path, const HTripleIndex<uint> grid_index) {

	char vert_name[__FILE_NAME_BUF_SIZE], face_name[__FILE_NAME_BUF_SIZE];
	string str;

	getVertPatchName(grid_index, vert_name);
	getFacePatchName(grid_index, face_name);

	if (dir_path) {
		str = dir_path;
		str += hPathSeperator();
	}
	str += vert_name;
	stringToCstr(str, vert_name);

	str.clear();
	if (dir_path) {
		str = dir_path;
		str += hPathSeperator();
	}
	str += face_name;
	stringToCstr(str, face_name);

	if(!HMeshPatch::openForWrite(vert_name, face_name))
		return false;
	return true;
}

bool HGridPatch::openForRead(const char* dir_path, const HTripleIndex<uint> grid_index) {

	char vert_name[__FILE_NAME_BUF_SIZE], face_name[__FILE_NAME_BUF_SIZE];
	string str;

	getVertPatchName(grid_index, vert_name);
	getFacePatchName(grid_index, face_name);

	if (dir_path) {
		str = dir_path;
		str += hPathSeperator();
	}
	str += vert_name;
	stringToCstr(str, vert_name);

	str.clear();
	if (dir_path) {
		str = dir_path;
		str += hPathSeperator();
	}
	str += face_name;
	stringToCstr(str, face_name);

	if(!HMeshPatch::openForRead(vert_name, face_name))
		return false;
	return true;
}