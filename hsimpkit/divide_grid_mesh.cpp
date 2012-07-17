#include "divide_grid_mesh.h"
#include "trivial.h"

bool HMeshGridDivide::readPly(char* ply_name, uint X, uint Y, uint Z) {

	PlyStream ply_stream;
	int i;
	HVertex v;

	if (!ply_stream.openForRead(ply_name)) {
		ostringstream oss << "open " << ply_name << " failed" << endl;
		info(oss);
		return false;
	}

	PlyFile *ply = ply_stream.plyFile();
	binary_file = false;

	if (ply->file_type == PLY_BINARY_BE || ply->file_type == PLY_BINARY_LE) {
		binary_file = true;
		if(fgetpos(ply->fp, &vert_start) != 0) {
			ostringstream oss << "get " << ply_name << " pos failed" << endl;
			info(oss);
			return false;
		}
	}
	else {
		string file_name = tmp_base;
		file_name += hPathSeperator();
		file_name += getFilename(ply_name) + ".vertbin";
		
		vertbin_name = new char[file_name.length() + 1];
		stringToCstr(file_name, vertbin_name);
		vertbin_out.open(vertbin_name, fstream::binary | fstream::out);
	}

	for (i = 0; i < ply_stream.getVertexCount(); i ++) {
		ply_stream.nextVertex(v);
		addVertexFirst(i, v);
	}
}