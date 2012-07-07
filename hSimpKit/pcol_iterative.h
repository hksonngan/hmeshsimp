/*
 *  iteratively perform the vertex pair collapse
 *
 *  author: ht
 */

#ifndef __H_ITERATIVE_PAIR_COLLAPSE__
#define __H_ITERATIVE_PAIR_COLLAPSE__ 

// for MxBlock
#define HAVE_CASTING_LIMITS

#include <algorithm>
#include "pcol_vertex.h"
#include "pcol_other_structures.h"
#include "util_common.h"
#include "h_dynamarray.h"
#include "MxHeap.h"

typedef CollapsablePair* pCollapsablePair;

inline bool pair_comp(const pCollapsablePair &pair1, const pCollapsablePair &pair2) {

	return (*pair1) < (*pair2);
};

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
	inline void addFace(HFace face);
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

	// collect start vertices from adjacent faces
	inline void collectStarVertices(uint vert_index, vert_arr *starVertices);
	inline void collapsePair(CollapsablePair *pair);
	inline void mergePairs(pair_arr &pairs1, pair_arr &pairs2, pair_arr &new_pairs);
	// guarantee vert1 < vert2 for all pairs in the arr
	inline void keepPairArrOrder(pair_arr &pairs);
	// change one vertex index to another for all pairs in the arr
	inline void changePairsOneVert(pair_arr &pairs);

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
	// set the new id, this is important!!
	cvert.setNewId(vertices.count());
	vertices.push_back(cvert);
}

void PairCollapse::addFace(HFace face) {

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
		cface = faces[cvert.adjacent_faces[i]];
		
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

void PairCollapse::keepPairArrOrder(pair_arr &pairs) {

	for (int i = 0; i < pairs.count(); i ++)
		pairs[i]->keepOrder();
}

void changePairsOneVert(pair_arr &pairs) {

	for (int i = 0; i < pairs.count(); i ++) {
		pairs[i]->keepOrder();
	}
}

void PairCollapse::mergePairs(pair_arr &pairs1, pair_arr &pairs2, pair_arr &new_pairs) {
	
	int i, j, k;

	// sort
	sort(pairs1.pointer(0), pairs1.pointer(pairs1.count() - 1), pair_comp);
	sort(pairs1.pointer(0), pairs1.pointer(pairs1.count() - 1), pair_comp);

	new_pairs.clear();
	new_pairs.resize(pairs1.count() + pairs2.count());

	// merge
	for (i = 0, j = 0, k = 0; i < pairs1.count() || j < pairs2.count(); k ++) {

		// skip invalid pairs
		//for (; i < pairs1.count() && 
		//	vertices[pairs1[i]->vert1].new_id == vertices[pairs1[i]->vert2].new_id;
		//	i ++) {
		//	
		//}
		//for (; j < pairs2.count() && 
		//	vertices[pairs2[j]->vert1].new_id == vertices[pairs2[j]->vert2].new_id;
		//	j ++) {
		//	
		//}

		// no repetition
		if (pairs1[i] == pairs2[i]) {
		}
		if (*pairs1[i] == *pairs2[j]) {
			new_pairs.push_back(pairs1[i]);
			pair_heap.remove(pairs2[j]);
			i ++; j ++;
		}
		else if (*pairs1[i] < *pairs2[j]) {
			new_pairs.push_back(pairs1[i]);
			pair_heap.update(pairs1[i]);
			i ++;
		}
		else {
			new_pairs.push_back(pairs2[j]);
			pair_heap.update(pairs2[j]);
			j ++;
		}
	}
}

void PairCollapse::collapsePair(CollapsablePair *pair) {

	int i;

	//collectStarVertices(pair->vert1, &starVerts1);
	//collectStarVertices(pair->vert2, &starVerts2);

	//for (i = 0; i < starVerts1.count(); i ++) {
	//	vertices[starVerts1[i]].flag = 0;
	//}
	//for (i = 0; i < starVerts2.count(); i ++) {
	//	vertices[starVerts2[i]].flag = 0;
	//}

	//for (i = 0; i < starVerts1.count(); i ++) {
	//	vertices[starVerts1[i]].flag ++;
	//}
	//for (i = 0; i < starVerts2.count(); i ++) {
	//	vertices[starVerts2[i]].flag ++;
	//}

	//if (pair->vert1 > pair->vert2) {
	//	hswap(pair->vert1, pair->vert2);
	//}

	// set the new_id field in order to invalidate
	// some faces and pairs, this may cause the 
	// order 'vert1 < vert2' broken
	vertices[pair->vert2].setNewId(pair->vert1);
	
	pair_arr &pairs1 = vertices[pair->vert1].adjacent_col_pairs;
	pair_arr &pairs2 = vertices[pair->vert2].adjacent_col_pairs;
	changePairsOneVert(pairs2);

	pair_arr new_pairs;
	mergePairs(pairs1, pairs2, new_pairs);
	hswap(pairs1, new_pairs);
	pairs2.freeSpace();
}

#endif //__H_ITERATIVE_PAIR_COLLAPSE__ 