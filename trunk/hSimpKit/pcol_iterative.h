/*
 *  iteratively perform the vertex pair collapse
 *
 *  author: ht
 */

#ifndef __H_ITERATIVE_PAIR_COLLAPSE__
#define __H_ITERATIVE_PAIR_COLLAPSE__ 

#include "pcol_vertex.h"
#include "pcol_other_structures.h"
#include "util_common.h"

class PairCollapse {
public:

	/////////////////////////////////////
	// initializers
	/////////////////////////////////////

	PairCollapse();
	// set the capacity for the container 
	// and allocate the memory space
	void allocVerts(uint _vert_count);
	void allocFaces(uint _face_count);
	// DO add vertices first and completely
	inline void addVertex(HVertex vert);
	inline void addFace(HTripleIndex face);
	// collect all valid pairs based on
	// specific measurement after the 
	// vertices and faces are ready
	// this function should be overrided
	// in specific derivative class
	void collectPairs() {}
	// this function should be overrided
	// in specific derivative class
	CollapsablePair* createPair(uint _vert1, uint _vert2) {}
	// add the pair to the heap and update the pair adjacent
	// information of the vertices
	inline void addCollapsablePair(CollapsablePair *new_pair);
	// init after the vertices and faces are ready
	void intialize();

	////////////////////////////////////
	// computing
	////////////////////////////////////
	
	// evaluate the target placement and error incurred,
	// and update the pair's content
	// this function should be overrided
	// in specific derivative class
	HVertex evaluatePair(CollapsablePair *pair) {}
	// simplify targeting vertex
	// this function should be overrided
	// in specific derivative class
	bool targetVert(uint targe_count) {}
	// simplify targeting face
	// this function should be overrided
	// in specific derivative class
	bool targetFace(uint targe_count) {}
	CollapsablePair* extractTopPair() { return (CollapsablePair *)pair_heap.extract(); }

	////////////////////////////////////
	// collapsing & linkage operation
	////////////////////////////////////

	inline void collectStarVertices(uint vert_index, vert_arr *starVertices);
	inline void collapsePair(CollapsablePair *pair);


	// clear heap
	void clear();

protected:
	HDynamArray<CollapsableVertex>	vertices;
	uint	valid_verts;
	HDynamArray<CollapseFace>	faces;
	uint	valid_faces;
	MxHeap	pair_heap;

	/////////////////////////////////////
	// assisting temporal variables
	/////////////////////////////////////

	static CollapsableVertex cvert;
	static CollapseFace cface;
	static vert_arr starVerts1, starVerts2;
};

void PairCollapse::addVertex(HVertex vert) {
	cvert.Set(vert.x, vert.y, vert.z);
	vertices.push_back(cvert);
}

void PairCollapse::addFace(HTripleIndex face) {

	cface.set(face.i, face.j, face.k);
	faces.push_back(cface);

	// add the face index to the vertices
	vertices[face.i].adjacent_faces.push_back(faces.count() - 1);
	vertices[face.j].adjacent_faces.push_back(faces.count() - 1);
	vertices[face.k].adjacent_faces.push_back(faces.count() - 1);
}

void PairCollapse::collectStarVertices(uint vert_index, vert_arr *starVertices) {

	starVertices->clear();
	cvert = vertices[vert_index];

	for (int i = 0; i < cvert.adjacent_faces.count(); i ++) {
		cface = cvert.adjacent_faces[i];
		
		if (cface.i != vert_index && !starVertices->exist(cface.i))
			starVertices->push_back(cface.i);
		if (cface.j != vert_index && !starVertices->exist(cface.j))
			starVertices->push_back(cface.j);
		if (cface.k != vert_index && !starVertices->exist(cface.k))
			starVertices->push_back(cface.k);
	}
}

void PairCollapse::addCollapsablePair(CollapsablePair *new_pair) {

	vertices[new_pair->vert1].adjacent_col_pairs.push_back(new_pair);
	vertices[new_pair->vert2].adjacent_col_pairs.push_back(new_pair);

	if (!new_pair->is_in_heap()) {
		pair_heap.insert(new_pair);
	}
}

void PairCollapse::collapsePair(CollapsablePair *pair) {

	int i;

	collectStarVertices(pair->vert1, &starVerts1);
	collectStarVertices(pair->vert2, &starVerts2);

	for (i = 0; i < starVerts1.count(); i ++) {
		vertices[starVerts1[i]].flag = 0;
	}
	for (i = 0; i < starVerts2.count(); i ++) {
		vertices[starVerts2[i]].flag = 0;
	}

	for (i = 0; i < starVerts1.count(); i ++) {
		vertices[starVerts1[i]].flag ++;
	}
	for (i = 0; i < starVerts2.count(); i ++) {
		vertices[starVerts2[i]].flag ++;
	}


}

#endif //__H_ITERATIVE_PAIR_COLLAPSE__ 