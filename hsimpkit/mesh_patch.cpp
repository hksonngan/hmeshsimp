#include "mesh_patch.h"

/* -- HMeshPatch -- */
bool HMeshPatch::openForWrite(const char* vert_name, const char* face_name) {

	uint n = 0;

	vert_file.open(vert_name, fstream::binary | fstream::out);
	if (!vert_file.good()) {
		cerr << "#ERROR: " << vert_name << " vert file open failed" << endl;
		return false;
	}
	// count of vertices
	vert_file.write((char *)&n, sizeof(uint));

	face_file.open(face_name, fstream::binary | fstream::out);
	if (!face_file.good()) {
		cerr << "#ERROR: " << face_name << " face file open failed" << endl;
		return false;
	}
	// count of faces
	vert_file.write((char *)&n, sizeof(uint));

	return true;
}

/* -- HGridPatch -- */
bool HGridPatch::openForWrite(HTripleIndex<uint> grid_index) {

	char vert_name[400], face_name[400];

	getVertPatchName(grid_index, vert_name);
	getFacePatchName(grid_index, face_name);

	if(!openForWrite(vert_name, face_name))
		return false;
	return true;
}