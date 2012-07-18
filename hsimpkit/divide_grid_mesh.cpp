#include "divide_grid_mesh.h"
#include "trivial.h"

/*
 *  The occupancy rate in fact never surpasses 1%
 *    -- [Shaffer. A Multi-resolution Representation for Massive Meshes]
 */
float HMeshGridDivide::MAX_OCCUPANCY = 0.1;

bool HMeshGridDivide::readPlyFirst(const char* _ply_name) {

	PlyStream ply_stream;
	int i;
	HVertex v;
	ostringstream oss;

	file_name = _ply_name;

	if (!ply_stream.openForRead(_ply_name)) {
		oss.clear();
		oss << "\t#ERROR: open " << _ply_name << " failed" << endl;
		info(oss);
		return false;
	}

	vert_count = ply_stream.getVertexCount();
	face_count = ply_stream.getFaceCount();

	PlyFile *ply = ply_stream.plyFile();

	/* open vertex binary file for write */
	string file_name = tmp_base;
	file_name += hPathSeperator();
	file_name += getFilename(_ply_name) + ".vertbin";
	
	vertbin_name = new char[file_name.length() + 1];
	stringToCstr(file_name, vertbin_name);
	vert_bin.openForWrite(vertbin_name);

	/* read the vertices */
	for (i = 0; i < ply_stream.getVertexCount(); i ++) {

		ply_stream.nextVertex(v);
		if (!addVertexFirst(i, v)) {
			oss.clear();
			oss << "\t#ERROR: write vertex " << i << " to vertex binary file failed" << endl;
			info(oss);
			return false;
		}


	}

	vert_bin.closeWriteFile();

	oss << "\t_______________________________________________" << endl
		<< "\tfirst pass complete reading" << endl
		<< "\tvertices:\t" << vert_count << "\tfaces:\t" << face_count << endl
		<< "\tbounding box:" << endl
		<< "\t\tx\t" << min_x << "\t" << max_x << endl
		<< "\t\ty\t" << min_y << "\t" << max_y << endl
		<< "\t\tz\t" << min_z << "\t" << max_z << endl << endl;

	info(oss);

	return true;
}

bool HMeshGridDivide::readPlySecond(uint _X, uint _Y, uint _Z) {

	PlyStream ply_stream;
	int i;
	HVertex v;
	ostringstream oss;

	x_div = _X;
	y_div = _Y;
	z_div = _Z;

	if (!ply_stream.openForRead(file_name)) {
		oss.clear();
		oss << "\t#ERROR: open " << file_name << " failed" << endl;
		info(oss);
		return false;
	}

	partitionInit();

	vert_bin.openForRead(vertbin_name);

	/* read the faces */

	return true;
}