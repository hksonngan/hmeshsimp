#include "divide_grid_mesh.h"
#include "trivial.h"

/*
 *  The occupancy rate in fact never surpasses 1%
 *    -- [Shaffer. A Multi-resolution Representation for Massive Meshes]
 */
float HMeshGridDivide::MAX_OCCUPANCY = 0.01;

bool HMeshGridDivide::readPlyFirst(const char* _ply_name) {

	PlyStream ply_stream;
	int i;
	HVertex v;
	ostringstream oss;
	HTripleIndex<uint> pi;
	HGridPatch* pPatch;

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

		getGridIndex(v, pi);
		pPatch = getPatch(pi);
		pPatch->addInteriorVertex(v);
	}

	vert_bin.closeWriteFile();
	ply_stream.close();

	oss << "\t_______________________________________________" << endl
		<< "\tfirst pass complete" << endl
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
	LRUVertex cache_v1, cache_v2, cache_v3;
	HVertex v;
	HTripleIndex<uint> face, pi;
	ostringstream oss;
	HGridPatch pPatch;

	x_div = _X;
	y_div = _Y;
	z_div = _Z;

	partitionInit();

	if (!ply_stream.openForRead(file_name)) {
		oss.clear();
		oss << "\t#ERROR: open " << file_name << " failed" << endl;
		info(oss);
		return false;
	}

	/* read the vertices */
	for (i = 0; i < ply_stream.getVertexCount(); i ++) {

		ply_stream.nextVertex(v);

		getGridIndex(v, pi);
		pPatch = getPatch(pi);
		pPatch->addInteriorVertex(v);
	}

	vert_bin.openForRead(vertbin_name);

	/* read the faces */
	for (i = 0; i < ply_stream.getFaceCount(); i ++) {

		ply_stream.nextFace(face);

		if (vert_bin.indexedRead(face.i, cache_v1) == LRU_FAIL) {
			oss << "\t#ERROR: read vertex binary file failed" << endl;
			info(oss);
			return false;
		}

		if (vert_bin.indexedRead(face.j, cache_v2) == LRU_FAIL) {
			oss << "\t#ERROR: read vertex binary file failed" << endl;
			info(oss);
			return false;
		}

		if (vert_bin.indexedRead(face.k, cache_v3) == LRU_FAIL) {
			oss << "\t#ERROR: read vertex binary file failed" << endl;
			info(oss);
			return false;
		}

		
	}

	return true;
}