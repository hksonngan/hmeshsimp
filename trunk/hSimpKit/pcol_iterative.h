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
#include <string.h>
#include "pcol_vertex.h"
#include "pcol_other_structures.h"
#include "util_common.h"
#include "h_dynamarray.h"
#include "MxHeap.h"

#define INFO_BUF_CAPACITY 2000

typedef CollapsablePair* pCollapsablePair;

static inline bool pair_comp(const pCollapsablePair &pair1, const pCollapsablePair &pair2) {

	return (*pair1) < (*pair2);
};

class FaceIndexComp {
public:
	bool operator(const uint &face_index1, const uint &face_index2) {

		return faces->elem(face_index1).unsequencedLessThan(faces->elem(face_index2));
	}

	void setFaces(HDynamArray<CollapsableFace> *_faces) { faces = _faces; }

private:
	HDynamArray<CollapsableFace> *faces;
};

class PairCollapse {
public:

	/////////////////////////////////////
	// initializers

	PairCollapse();
	// set the capacity for the container 
	// and allocate the memory space
	void allocVerts(uint _vert_count);
	void allocFaces(uint _face_count);
	// DO add vertices first and completely
	inline void addVertex(HVertex vert);
	inline bool addFace(HFace face);
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

	// collect start vertices from adjacent faces
	inline void collectStarVertices(uint vert_index, vert_arr *starVertices);
	inline void collapsePair(pCollapsablePair &pair);
	// guarantee vert1 < vert2 for all pairs in the arr
	inline void keepPairArrOrder(pair_arr &pairs);
	// change one vertex index to another for all pairs in the arr
	inline void changePairsOneVert(pair_arr &pairs,  uint orig, uint dst);
	// update valid pair in the heap and remove invalid pair
	inline void updateValidPairOrRemove(pCollapsablePair &pair, pair_arr &new_pairs);
	inline void mergePairs(pair_arr &pairs1, pair_arr &pairs2, pair_arr &new_pairs);
	// change one vertex index to another for all faces in the arr
	inline void changeFacesOneVert(face_arr &face_indices, uint orig, uint dst);
	inline void mergeFaces(face_arr &faces1, face_arr &faces2, face_arr &new_faces);

	///////////////////////////////////////
	// other than simplification

	inline void addInfo(char *s);

	// clear heap
	void clear();

protected:
	HDynamArray<CollapsableVertex>	vertices;
	uint	valid_verts;
	HDynamArray<CollapsableFace>	faces;
	uint	valid_faces;
	MxHeap	pair_heap;

	FaceIndexComp faceIndexComp;

	char	INFO_BUF[INFO_BUF_CAPACITY];
	uint	info_buf_size;

	/////////////////////////////////////
	// assisting temporal variables

	static CollapsableVertex cvert;
	static CollapsableFace cface;
	static vert_arr starVerts1, starVerts2;
};

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

void changePairsOneVert(pair_arr &pairs, uint orig, uint dst) {

	for (int i = 0; i < pairs.count(); i ++) {
		pairs[i]->changeOneVert(orig, dst);
		pairs[i]->keepOrder();
	}
}

void PairCollapse::updateValidPairOrRemove(pCollapsablePair &pair, pair_arr &new_pairs) {

	if (pair->valid()) {
		new_pairs.push_back(pair);
		pair_heap.update(pair);
	}
	else {
		pair_heap.remove(pair);
		delete[] pair;
	}
}

void PairCollapse::mergePairs(pair_arr &pairs1, pair_arr &pairs2, pair_arr &new_pairs) {
	
	int i, j;

	// Sort in case of non-duplicated merge
	// When sorting the pair_arr, the array is sorted based on the two vertices
	// indices (you may refer to the definition of class 'CollapsablePair' and
	// function 'pair_comp'). So there may result in two types of equals (equal
	// elements reside continuously in the sorted array), one is the value equal,
	// which means that they point to different structs which have the same value
	// in vert1, vert2 field, one is pointer equal which means the two different
	// pointer points to the same struct. So these may need some special treatment
	// when merging
	sort(pairs1.pointer(0), pairs1.pointer(pairs1.count() - 1), pair_comp);
	sort(pairs2.pointer(0), pairs2.pointer(pairs2.count() - 1), pair_comp);

	new_pairs.clear();
	new_pairs.resize(pairs1.count() + pairs2.count());

	// merge
	for (i = 0, j = 0; i < pairs1.count() || j < pairs2.count();) {

		// pairs[i] and pairs[j] points to the same struct
		if (i < pairs1.count() && j < pairs2.count() && pairs1[i] == pairs2[j]) {
			
			//updateValidPairOrRemove(pairs1[i], new_pairs);

			// this is the collapsed pair
			pair_heap.remove(pairs1[i]);
			delete[] pairs1[i];
			i ++; j ++;
		}
		// pairs[i] and pairs[j] points to different value equal structs
		else if (i < pairs1.count() && j < pairs2.count() && *pairs1[i] == *pairs2[j]) {
			
			//updateValidPairOrRemove(pairs1[i], new_pairs);

			// these are the duplicated pairs
			// update arbitrary one and remove another 
			new_pairs.push_back(pairs1[i]);
			pair_heap.update(pairs1[i]);
			pair_heap.remove(pairs2[j]);
			delete[] pairs2[j];
			i ++; j ++;
		}
		else if (j >= pairs2.count() || i < pairs1.count() && *pairs1[i] < *pairs2[j]) {

			//updateValidPairOrRemove(pairs1[i], new_pairs);

			new_pairs.push_back(pairs1[i]);
			pair_heap.update(pairs1[i]);
			i ++;
		}
		else {
			//updateValidPairOrRemove(pairs2[j], new_pairs);

			new_pairs.push_back(pairs2[j]);
			pair_heap.update(pairs2[j]);
			j ++;
		}
	}
}

void PairCollapse::changeFacesOneVert(face_arr &face_indices, uint orig, uint dst) {
	
	for (int i = 0; i < face_indices.count(); i ++) 
		faces[face_indices[i]].changeOneVert(orig, dst);
}

void PairCollapse::mergeFaces(face_arr &faces1, face_arr &faces2, face_arr &new_faces) {

	sort(faces1.pointer(0), faces1.pointer(faces1.count()), faceIndexComp);
	sort(faces2.pointer(0), faces2.pointer(faces2.count()), faceIndexComp);

	new_faces.clear();
	new_faces.resize(faces1.count() + faces2.count());

	int i, j;

	for (i = 0, j = 0; i < faces1.count() || j < faces2.count(); ) {

	}
}

void PairCollapse::collapsePair(pCollapsablePair &pair) {

	int i;

	// set the new_id field in order to invalidate
	// vert2 as well as some faces and pairs, this 
	// may cause the order 'vert1 < vert2' broken
	vertices[pair->vert2].setNewId(pair->vert1);

	pair_arr &pairs1 = vertices[pair->vert1].adjacent_col_pairs;
	pair_arr &pairs2 = vertices[pair->vert2].adjacent_col_pairs;
	changePairsOneVert(pairs2, pair->vert2, pair->vert1);

	pair_arr new_pairs;
	mergePairs(pairs1, pairs2, new_pairs);
	hswap(pairs1, new_pairs);
	pairs2.freeSpace();

	face_arr &faces1 = vertices[pair->vert1].adjacent_faces;
	face_arr &faces2 = vertices[pair->vert2].adjacent_faces;
	changeFacesOneVert(faces2, pair->vert2, pair->vert1);

	face_arr new_faces;
	mergeFaces(faces1, faces2, new_faces);
	hswap(faces1, new_faces);
	faces2.freeSpace();
}

///////////////////////////////////////////////////////////////
// operations other than simplification

void PairCollapse::addInfo(char *s) {

	memcpy(INFO_BUF + info_buf_size, s, strlen(s));
}

#endif //__H_ITERATIVE_PAIR_COLLAPSE__ 