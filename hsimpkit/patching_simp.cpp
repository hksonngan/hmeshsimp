#include "patching_simp.h"
#include "trivial.h"
#include "os_dependent.h"
#include "ecol_iterative_quadric.h"
#include "mem_stream.h"
#include "id_map_stream.h"
#include "io_common.h"


const float PatchingSimp::MAX_OCCUPANCY = 0.8;

bool PatchingSimp::tmpBase(char *s) { 

	tmp_base = s; 
	return hCreateDir(s);
}

bool PatchingSimp::readPlyFirst(char* _ply_name) {

	PlyStream ply_stream;
	int i;
	HVertex v;
	HTriple<uint> pi;
	HGridPatch* pPatch;

	file_name = _ply_name;
	temp_time.setStartPoint();

	if (!ply_stream.openForRead(_ply_name)) {
		INFO << "#ERROR: open " << _ply_name << " failed" << endl;
		cerr << "#ERROR: open " << _ply_name << " failed" << endl;
		return false;
	}

	vert_count = ply_stream.getVertexCount();
	face_count = ply_stream.getFaceCount();

	/* open vertex binary file for write */
	string str_vertbin_name;
	if (tmp_base) {
		str_vertbin_name = tmp_base;
		str_vertbin_name += hPathSeperator();
	}
	str_vertbin_name += getFilename(_ply_name) + ".vertbin";
	
	vertbin_name = new char[str_vertbin_name.length() + 1];
	stringToCstr(str_vertbin_name, vertbin_name);
	vert_bin.openForWrite(vertbin_name);

	/* read the vertices */
	for (i = 0; i < ply_stream.getVertexCount(); i ++) {

		ply_stream.nextVertex(v);
		if (!addVertexFirst(i, v)) {
			INFO << "#ERROR: write vertex " << i << " to vertex binary file failed" << endl;
			cerr << "#ERROR: write vertex " << i << " to vertex binary file failed" << endl;
			return false;
		}
	}

	vert_bin.closeWriteFile();
	ply_stream.close();
	temp_time.setEndPoint();
	total_time += temp_time;

	INFO << "\t-----------------------------------------------" << endl
		<< "\tfirst pass complete" << endl
		<< "\tvertices:\t" << vert_count << "\tfaces:\t" << face_count << endl
		<< "\tbounding box:" << endl
		<< "\t\tx\t" << min_x << "\t" << max_x << endl
		<< "\t\ty\t" << min_y << "\t" << max_y << endl
		<< "\t\tz\t" << min_z << "\t" << max_z << endl
		<< "\ttime consuming: " << temp_time << endl;

	cout << "\t-----------------------------------------------" << endl
		<< "\tfirst pass complete" << endl
		<< "\tvertices:\t" << vert_count << "\tfaces:\t" << face_count << endl
		<< "\tbounding box:" << endl
		<< "\t\tx\t" << min_x << "\t" << max_x << endl
		<< "\t\ty\t" << min_y << "\t" << max_y << endl
		<< "\t\tz\t" << min_z << "\t" << max_z << endl
		<< "\ttime consuming: " << temp_time << endl;

	return true;
}

bool PatchingSimp::readPlySecond(uint _X, uint _Y, uint _Z) {

	PlyStream ply_stream;
	int i;
	LRUVertex cache_v1, cache_v2, cache_v3;
	HVertex v;
	HTriple<uint> face, pi;
	HGridPatch *pPatch;

	x_div = _X;
	y_div = _Y;
	z_div = _Z;

	partitionInit();

	if (!ply_stream.openForRead(file_name)) {
		INFO << "#ERROR: open " << file_name << " failed" << endl;
		cerr << "#ERROR: open " << file_name << " failed" << endl;
		return false;
	}

	/* read the vertices */
	for (i = 0; i < ply_stream.getVertexCount(); i ++) {
		ply_stream.nextVertex(v);

		getGridIndex(v, pi);
		pPatch = getPatch(pi);
		if (!pPatch->addInteriorVertex(i, v))
			return false;
	}

	if (vert_bin.openForRead(vertbin_name) == LRU_FAIL) {
		INFO << "#ERROR: open " << vertbin_name << " failed" << endl;
		cerr << "#ERROR: open " << vertbin_name << " failed" << endl;
		return false;
	}

	/* read the faces */
	for (i = 0; i < ply_stream.getFaceCount(); i ++) {

		ply_stream.nextFace(face);

		if (vert_bin.indexedRead(face.i, cache_v1) == LRU_FAIL) {
			INFO << "#ERROR: read vertex binary file failed" << endl;
			cerr << "#ERROR: read vertex binary file failed" << endl;
			return false;
		}

		if (vert_bin.indexedRead(face.j, cache_v2) == LRU_FAIL) {
			INFO << "#ERROR: read vertex binary file failed" << endl;
			cerr << "#ERROR: read vertex binary file failed" << endl;
			return false;
		}

		if (vert_bin.indexedRead(face.k, cache_v3) == LRU_FAIL) {
			INFO << "#ERROR: read vertex binary file failed" << endl;
			cerr << "#ERROR: read vertex binary file failed" << endl;
			return false;
		}

		addFaceToPatch(face, cache_v1.v, cache_v2.v, cache_v3.v);		
	}

	return partitionEnd();
}

bool PatchingSimp::patchesToPly() {

	int i;
	HGridPatch patch;

	for (i = 0; i < patchIndices.count(); i ++) {
		if (!patch.patchToPly(tmp_base, patchIndices[i])) {
			cerr << "#ERROR: write patch " << patchIndices[i].i << "_" << patchIndices[i].j
				<< "_" << patchIndices[i].k << " to ply failed" << endl;
			return false;
		}
	}

	return true;
}

bool PatchingSimp::simplfiyPatchesToPly(uint target_vert) {
	
	int i;
	HGridPatch patch;

	for (i = 0; i < patchIndices.count(); i ++) {
		if (!patch.pairCollapseToPly(tmp_base, patchIndices[i], vert_count, target_vert)) {
			cerr << "#ERROR: simplifying patch " << patchIndices[i].i << "_" << patchIndices[i].j 
				<< "_" << patchIndices[i].k << " to ply failed" << endl;
			return false;
		}
	}

	return true;
}

bool PatchingSimp::mergeSimpPly(uint target_vert, bool binary) {

	int i;
	HGridPatch patch;
	mstream<HTriple<uint>> face_stream;
	mstream<HVertex> vert_stream;
	/* the map between original id of interior 
	 * boundary vertices and the output id */
	IdMapMStream bound_id_map;
	uint init_buckets;

	temp_time.start();

	bound_id_map.map.rehash(ibt.faceCount() / 2);
	init_buckets = bound_id_map.map.bucket_count();

	if (!mergeSimp(target_vert, vert_stream, face_stream, bound_id_map)) 
		return false;

	INFO << "\tbound id map, init buckets: " << init_buckets << " final buckets: " << bound_id_map.map.bucket_count() << endl
		<< "\tavg nodes per bucket: " << bound_id_map.map.load_factor() << " max nodes: " << bound_id_map.map.max_load_factor() << endl;

	cout << "\tbound id map, init buckets: " << init_buckets << " final buckets: " << bound_id_map.map.bucket_count() << endl
		<< "\tavg nodes per bucket: " << bound_id_map.map.load_factor() << " max nodes: " << bound_id_map.map.max_load_factor() << endl;

	string out_name;
	if (tmp_base) {
		out_name = tmp_base;
		out_name += hPathSeperator();
	}
	if (binary)
#ifdef WRITE_PATCH_BINARY
		out_name += getFilename(file_name) + "_psimp_bin.ply";
#else
		out_name += getFilename(file_name) + "_psimp_txt.ply";
#endif
	else
		out_name += getFilename(file_name) + "_psimp_txt.ply";


	ofstream fout;
	if (binary)
		fout.open(out_name.c_str(), fstream::out | fstream::binary);
	else
		fout.open(out_name.c_str(), fstream::out);

	/* write head */
	fout << "ply" << endl;
	if (binary)
#ifdef WRITE_PATCH_BINARY
		fout << "format " << getPlyBinaryFormat() << endl;
#else
		fout << "format ascii 1.0" << endl;
#endif
	else
		fout << "format ascii 1.0" << endl;
	fout << "comment generated by patching simp" << endl;

	fout << "element vertex " << simp_verts << endl;
	fout << "property float x" << endl;
	fout << "property float y" << endl;
	fout << "property float z" << endl;
	fout << "element face " << simp_faces << endl;
	fout << "property list uchar int vertex_indices" << endl;
	fout << "end_header" << endl;

	for (i = 0; i < vert_stream.count(); i ++) {
		if (binary)
			write_vert(fout, vert_stream[i]);
		else
			write_vert_txt(fout, vert_stream[i]);
		if (!fout.good()) {
			cerr << "#ERROR: writing vertex to the simplified mesh failed" << endl;
			return false;
		}
	}
	
	for (i = 0; i < face_stream.count(); i ++) {
		if (binary)
			write_face(fout, face_stream[i]);
		else
			write_face_txt(fout, face_stream[i]);
		if (!fout.good()) {
			cerr << "#ERROR: writing face for to simplified mesh failed" << endl;
			return false;
		}
	}

	temp_time.end();
	INFO << "\ttime consuming: " << temp_time << endl;
	cout << "\ttime consuming: " << temp_time << endl;

	return true;
}
