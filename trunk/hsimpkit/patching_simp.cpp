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
	ostringstream oss;
	HTriple<uint> pi;
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
			oss.clear();
			oss << "\t#ERROR: write vertex " << i << " to vertex binary file failed" << endl;
			info(oss);
			return false;
		}
	}

	vert_bin.closeWriteFile();
	ply_stream.close();

	oss << "\t_______________________________________________" << endl
		<< "\tfirst pass complete" << endl
		<< "\tvertices:\t" << vert_count << "\tfaces:\t" << face_count << endl
		<< "\tbounding box:" << endl
		<< "\t\tx\t" << min_x << "\t" << max_x << endl
		<< "\t\ty\t" << min_y << "\t" << max_y << endl
		<< "\t\tz\t" << min_z << "\t" << max_z << endl;

	info(oss);

	return true;
}

bool PatchingSimp::readPlySecond(uint _X, uint _Y, uint _Z) {

	PlyStream ply_stream;
	int i;
	LRUVertex cache_v1, cache_v2, cache_v3;
	HVertex v;
	HTriple<uint> face, pi;
	ostringstream oss;
	HGridPatch *pPatch;

	x_div = _X;
	y_div = _Y;
	z_div = _Z;

	partitionInit();

	if (!ply_stream.openForRead(file_name)) {
		oss.clear();
		oss << "#ERROR: open " << file_name << " failed" << endl;
		info(oss);
		return false;
	}

	/* read the vertices */
	for (i = 0; i < ply_stream.getVertexCount(); i ++) {

		if (i == 30) {
			int k = 0;
			k ++;
		}

		ply_stream.nextVertex(v);

		float f;
		f = v.x;
		switchBytes((char*)&f, sizeof(float));
		f = v.y;
		switchBytes((char*)&f, sizeof(float));
		f = v.z;
		switchBytes((char*)&f, sizeof(float));

		getGridIndex(v, pi);
		pPatch = getPatch(pi);

		if (!pPatch->addInteriorVertex(i, v))
			return false;
	}

	if (vert_bin.openForRead(vertbin_name) == LRU_FAIL) {
		oss.clear();
		oss << "#ERROR: open " << vertbin_name << " failed" << endl;
		info(oss);
		return false;
	}

	/* read the faces */
	for (i = 0; i < ply_stream.getFaceCount(); i ++) {

		ply_stream.nextFace(face);

		if (vert_bin.indexedRead(face.i, cache_v1) == LRU_FAIL) {
			oss.clear();
			oss << "#ERROR: read vertex binary file failed" << endl;
			info(oss);
			return false;
		}

		if (vert_bin.indexedRead(face.j, cache_v2) == LRU_FAIL) {
			oss.clear();
			oss << "#ERROR: read vertex binary file failed" << endl;
			info(oss);
			return false;
		}

		if (vert_bin.indexedRead(face.k, cache_v3) == LRU_FAIL) {
			oss.clear();
			oss << "#ERROR: read vertex binary file failed" << endl;
			info(oss);
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
	uint verts_gen;
	mstream<HTriple<uint>> face_stream;
	mstream<HTriple<uint>> ib_faces;
	mstream<HVertex> vert_stream;
	HTriple<uint> face;
	/* the map between original id of interior 
	 * boundary vertices and the output id */
	IdMapMStream bound_id_map;

	bound_id_map.map.rehash(ibt.faceCount() * 3 / 4);

	if (!mergeSimp(target_vert, vert_stream, face_stream, bound_id_map)) 
		return false;

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

	simp_verts = vert_stream.count();
	simp_faces = face_stream.count();

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

	return true;
}

template<class VOutType, class FOutType, class IdMapStreamType>
bool PatchingSimp::mergeSimp(uint target_vert, VOutType &vout, FOutType &fout, IdMapStreamType &bound_id_map) {

	int i;
	HGridPatch patch;
	uint verts_gen;
	mstream<HTriple<uint>> ib_faces;
	HTriple<uint> face;

	simp_verts = 0;
	for (i = 0; i < patchIndices.count(); i ++) {
		if (!patch.pairCollapse(tmp_base, patchIndices[i], simp_verts, vert_count, 
			target_vert, vout, fout, bound_id_map, verts_gen)) {
				cerr << "#ERROR: simplifying patch " << patchIndices[i].i << "_" << patchIndices[i].j 
					<< "_" << patchIndices[i].k << " to merge failed" << endl;
				return false;
		}
		simp_verts += verts_gen;
	}

	/* process the interior boundary triangles */
	if (!ibt.openIBTFileForRead(tmp_base))
		return false;

	ib_faces.resize(ibt.faceCount() / 4);
	for (i = 0; i < ibt.faceCount(); i ++) {
		if (!ibt.nextIBTriangle(face))
			return false;

		face.i = bound_id_map[face.i];
		face.j = bound_id_map[face.j];
		face.k = bound_id_map[face.k];
		if (face.i != face.j && face.i != face.k  && face.j != face.k)
			ib_faces.add(face);
	}
	ibt.closeIBTFileForRead();

	if (ib_faces.count() > 0)
		sort(ib_faces.pointer(0), ib_faces.pointer(ib_faces.count() - 1), face_comp);

	for (i = 0; i < ib_faces.count(); ) {
		face = ib_faces[i];
		fout.add(face);
		if (!fout.good()) {
			cerr << "#ERROR: adding ibtriangle to simplified face stream failed" << endl;
			return false;
		}
		// ignore duplication
		for (; i < ib_faces.count() && ib_faces[i] == face; i ++);
	}

	return true;
}
