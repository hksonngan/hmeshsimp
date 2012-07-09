#include "pcol_iterative.h"

#include <iostream>
#include <sstream>
#include <string>
#include "h_time.h"
#include "ply_stream.h"

using std::ostringstream;
using std::endl;

CollapsableVertex PairCollapse::cvert;
CollapsableFace PairCollapse::cface;
vert_arr PairCollapse::starVerts1;
vert_arr PairCollapse::starVerts2;

PairCollapse::PairCollapse() {
	info_buf_size = 0;
	faceIndexComp.setFaces(&faces);
	flog.open("ec.log");
}

PairCollapse::~PairCollapse() {
	flog.close();
}

void PairCollapse::allocVerts(uint _vert_count) {
	vertices.resize(_vert_count);
}

void PairCollapse::allocFaces(uint _face_count) {
	faces.resize(_face_count);
}

void PairCollapse::addVertex(HVertex vert) {
	cvert.Set(vert.x, vert.y, vert.z);
	// set the new id, this is important!!
	cvert.setNewId(vertices.count());
	vertices.push_back(cvert);
}

bool PairCollapse::addFace(HFace face) {

	cface.set(face.i, face.j, face.k);

	if (!cface.valid()) {
		addInfo("#error: duplicate vertices in input face\n");
		return false;
	}
	if (!cface.indicesInRange(0, vertices.count() - 1)) {
		addInfo("#error: vertex out of range in input face\n");
		return false;
	}

	faces.push_back(cface);

	// add the face index to the vertices
	vertices[face.i].adjacent_faces.push_back(faces.count() - 1);
	vertices[face.j].adjacent_faces.push_back(faces.count() - 1);
	vertices[face.k].adjacent_faces.push_back(faces.count() - 1);

	return true;
}

void PairCollapse::intialize() {
	valid_verts = vertices.count();
	valid_faces = faces.count();
	collectPairs();
}

void PairCollapse::collapsePair(pCollapsablePair &pair) {

	int i;

	// set the new_id field and new_vertex field in order 
	// to invalidate vert2 and maintain the collapse footprint
	vertices[pair->vert2].setNewId(pair->vert1);
	//vertices[pair->vert2].new_vertex.Set(pair->new_vertex);
	// vert1 will be the collapsed vertex, set to the new position
	vertices[pair->vert1].Set(pair->new_vertex);
	valid_verts --;

	pair_arr &pairs1 = vertices[pair->vert1].adjacent_col_pairs;
	pair_arr &pairs2 = vertices[pair->vert2].adjacent_col_pairs;
	// change the index of vert2 to vert1 for all pairs adjacent
	// to vert2, this may cause the order 'vert1 < vert2' broken
	// and some pairs to be invalid or duplicated
	changePairsOneVert(pairs2, pair->vert2, pair->vert1);

	pair_arr new_pairs;
	mergePairs(pairs1, pairs2, new_pairs);
	reevaluatePairs(new_pairs);
	hswap(pairs1, new_pairs);
	pairs2.freeSpace();

	face_arr &faces1 = vertices[pair->vert1].adjacent_faces;
	face_arr &faces2 = vertices[pair->vert2].adjacent_faces;
	// change the index of vert2 to vert1 for all faces adjacent
	// to vert2, this may cause some faces to be invalid or duplicated
	changeFacesOneVert(faces2, pair->vert2, pair->vert1);

	face_arr new_faces;
	mergeFaces(faces1, faces2, new_faces);
	hswap(faces1, new_faces);
	faces2.freeSpace();
}

bool PairCollapse::targetFace(uint target_count) {
	
	CollapsablePair* top_pair;

	HTime htime;

	while(valid_faces > target_count) {

		top_pair = (CollapsablePair *)pair_heap.extract();
		collapsePair(top_pair);
	}

	flog << "\t-----------------------------------------------" << endl
		<< "\tmodel simplified" << endl
		<< "\tvertex count:\t" << valid_verts << "\tface count:\t" << valid_faces << endl
		<< "\ttime consuming:\t" << htime.printElapseSec() << endl << endl;

	cout << "\t-----------------------------------------------" << endl
		<< "\tmodel simplified" << endl
		<< "\tvertex count:\t" << valid_verts << "\tface count:\t" << valid_faces << endl
		<< "\ttime consuming:\t" << htime.printElapseSec() << endl << endl;

	return true;
}

bool PairCollapse::readPly(char* filename) {

	PlyStream plyStream;
	int i;
	HVertex v;
	HFace f;
	HTime htime;

	if (!plyStream.openForRead(filename))
		return false;

	this->allocVerts(plyStream.getVertexCount());
	this->allocFaces(plyStream.getFaceCount());

	for (i = 0; i < plyStream.getVertexCount(); i ++) {

		if (!plyStream.nextVertex(v)) 
			return false;
		addVertex(v);
	}

	for (i = 0; i < plyStream.getFaceCount(); i ++) {

		if (!plyStream.nextFace(f)) 
			return false;
		addFace(f);
	}

	intialize();

	flog << "\t-----------------------------------------------" << endl
		<< "\tread file successfully" << endl
		<< "\tfile name:\t" << filename << endl
		<< "\tvertex count:\t" << plyStream.getVertexCount() << "\tface count:\t" << plyStream.getFaceCount() << endl
		<< "\tread file time:\t" << htime.printElapseSec() << endl << endl;

	cout << "\t-----------------------------------------------" << endl
		<< "\tread file successfully" << endl
		<< "\tfile name:\t" << filename << endl
		<< "\tvertex count:\t" << plyStream.getVertexCount() << "\tface count:\t" << plyStream.getFaceCount() << endl
		<< "\tread file time:\t" << htime.printElapseSec() << endl << endl;
}