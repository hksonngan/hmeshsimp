#include "pcol_iterative.h"

CollapsableVertex PairCollapse::cvert;
CollapsableFace PairCollapse::cface;
vert_arr PairCollapse::starVerts1;
vert_arr PairCollapse::starVerts2;

PairCollapse::PairCollapse() {
	info_buf_size = 0;
	faceIndexComp.setFaces(&faces);
}

void PairCollapse::allocVerts(uint _vert_count) {
	vertices.resize(_vert_count);
}

void PairCollapse::allocFaces(uint _face_count) {
	faces.resize(_face_count);
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

	while(valid_faces > target_count) {

		top_pair = (CollapsablePair *)pair_heap.extract();

		collapsePair(top_pair);
	}

	return true;
}