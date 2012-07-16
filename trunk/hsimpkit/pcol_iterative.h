/*
 *  Iteratively perform the vertex pair collapse
 *
 *  Author: Ht
 *  Email : waytofall916@gmail.com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */

#ifndef __H_ITERATIVE_PAIR_COLLAPSE__
#define __H_ITERATIVE_PAIR_COLLAPSE__ 

// for MxBlock
#define HAVE_CASTING_LIMITS

#include <algorithm>
#include <string.h>
#include <iostream>
#include <fstream>
#include "pcol_vertex.h"
#include "pcol_other_structures.h"
#include "util_common.h"
#include "h_dynamarray.h"
#include "MixKit/MxHeap.h"
#include "h_aug_time.h"

using std::ofstream;
using std::cout;
using std::endl;

#define INFO_BUF_CAPACITY 4000

typedef CollapsablePair* pCollapsablePair;

static inline bool pair_comp(const pCollapsablePair &pair1, const pCollapsablePair &pair2) {

	// the pointer in 'adjacent_col_pairs' may contain NULL pointer
	// let the NULL pointers go to the end of sorted array
	if (pair1 == NULL) 
		return false;
	else if (pair2 == NULL)
		return true;

	return (*pair1) < (*pair2);
};

class FaceIndexComp {
public:
	bool operator() (const uint &face_index1, const uint &face_index2) {

		// let invalid faces go the end
		//if (!faces->elem(face_index1).valid())
		//	return false;
		//else (!faces->elem(face_index2).valid())
		//	return true;

		return faces->elem(face_index1).unsequencedLessThan(faces->elem(face_index2));
	}

	void setFaces(HDynamArray<CollapsableFace> *_faces) { faces = _faces; }

private:
	HDynamArray<CollapsableFace> *faces;
};

class PairCollapse {
public:

	/////////////////////////////////////
	// Initializers

	PairCollapse();
	~PairCollapse();
	// set the capacity for the container 
	// and allocate the memory space
	virtual void allocVerts(uint _vert_count);
	virtual void allocFaces(uint _face_count);
	// DO add vertices first and completely
	virtual void addVertex(HVertex vert);
	virtual bool addFace(HFace face);
	// collect all valid pairs based on
	// specific measurement after the 
	// vertices and faces are ready
	// this function should be overrided
	// in specific derivative class
	virtual void collectPairs() = 0;
	// add the pair to the heap and update the pair adjacent
	// information of the vertices
	inline void addCollapsablePair(CollapsablePair *new_pair);
	// init after the vertices and faces are ready
	virtual void intialize();
	inline void unreferVertsCheck();
	
	
	////////////////////////////////////
	// Computing
	
	// simplify targeting vertex
	// this function should be overrided
	// in specific derivative class
	bool targetVert(uint targe_count);
	// simplify targeting face
	// this function should be overrided
	// in specific derivative class
	bool targetFace(uint targe_count);


	////////////////////////////////////
	// Collapsing & Linkage operation

	// collect start vertices from adjacent faces
	inline void collectStarVertices(uint vert_index, vert_arr *starVertices);
	// collect the faces of the edge <vert1, vert2>
	inline void collectEdgeFaces(uint vert1, uint vert2, face_arr &_faces);
	
	// evaluate the target placement and error incurred,
	// and update the pair's content
	// this function should be overrided
	// in specific derivative class
	virtual HVertex evaluatePair(CollapsablePair *pair) = 0;
	// !!an important function
	virtual void collapsePair(pCollapsablePair &pair);

	// guarantee vert1 < vert2 for all pairs in the arr
	//inline void keepPairArrOrder(pair_arr &pairs);
	// change one vertex index to another for all pairs in the arr
	inline void changePairsOneVert(pair_arr &pairs,  uint orig, uint dst);
	// push valid pair to the new array and remove invalid pair from heap
	inline void pushValidPairOrRemove(pCollapsablePair &pair, pair_arr &new_pairs);
	// set one pair null in the 'adjacent_col_pairs' array
	// for the vert
	inline void setOnePairNull(uint vert, pCollapsablePair pair);
	// check invalid and duplicated pairs and discard them, leaving
	// the valid and unique pairs into the new pair array
	inline void mergePairs(uint vert1, uint vert2);
	inline void reevaluatePairs(pair_arr &pairs);

	// change one vertex index to another for all faces in the arr
	inline void changeFacesOneVert(face_arr &face_indices, uint orig, uint dst);
	inline void mergeFaces(uint vert1, uint vert2);
	inline void markFaces(face_arr &_faces, unsigned char m);
	inline void collectMarkFaces(face_arr &faces_in, face_arr &faces_out, unsigned char m);


	///////////////////////////////////////
	// File I/O & Output Generation

	bool readPly(char* filename);
	bool writePly(char* filename);
	void generateOutputId();
	// for debug
	void outputIds(char* filename);


	///////////////////////////////////////
	// Other Than Simplification

	void addInfo(const char *s);
	char* getInfo() { return INFO_BUF; };
	void clearInfo() { info_buf_len = 0; INFO_BUF[0] = '\0'; };
	void totalTime();
	uint vertexCount() const { return vertices.count(); }
	uint faceCount() const { return faces.count(); }

	// clear heap
	void clear();

protected:
	HDynamArray<CollapsableVertex>	vertices;
	uint	valid_verts;
	HDynamArray<CollapsableFace>	faces;
	uint	valid_faces;
	MxHeap	pair_heap;

	uint	valid_vert_count;

	FaceIndexComp	faceIndexComp;

	char	INFO_BUF[INFO_BUF_CAPACITY];
	uint	info_buf_len;

	HAugTime read_time, run_time, write_time;

	/////////////////////////////////////
	// constants
	static const uint	DFLT_STAR_FACES = 6;
	static const uint	DFLT_STAR_PAIRS = 6;

	/////////////////////////////////////
	// assisting temporal variables
	CollapsableVertex	cvert;
	CollapsableFace	cface;
	vert_arr	starVerts1, starVerts2;
};

void PairCollapse::unreferVertsCheck() {
	
	valid_verts = 0;

	for (int i = 0; i < vertices.count(); i ++) 
		if (vertices[i].valid(i)) 
			valid_verts ++;
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

void PairCollapse::changePairsOneVert(pair_arr &pairs, uint orig, uint dst) {

	for (int i = 0; i < pairs.count(); i ++) 
		if (pairs[i]) {
			pairs[i]->changeOneVert(orig, dst);
			pairs[i]->keepOrder();
		}
}

void PairCollapse::pushValidPairOrRemove(pCollapsablePair &pair, pair_arr &new_pairs) {

	if (pair->valid()) {
		new_pairs.push_back(pair);
		//pair_heap.update(pair);
	}
	else {
		pair_heap.remove(pair);
		delete[] pair;
	}
}

void PairCollapse::setOnePairNull(uint vert, pCollapsablePair pair) {
	
	if (vert >= 0 && vert < vertices.count()) {

		pair_arr &pairs = vertices[vert].adjacent_col_pairs;
		for	(int i = 0; i < pairs.count(); i ++)
			if (pairs[i] == pair) 
				pairs[i] = NULL;
	}
}

void PairCollapse::mergePairs(uint vert1, uint vert2) {

	int i, j;

	/* pre process */

	pair_arr &pairs1 = vertices[vert1].adjacent_col_pairs;
	pair_arr &pairs2 = vertices[vert2].adjacent_col_pairs;
	// change the index of vert2 to vert1 for all pairs adjacent
	// to vert2, this may cause the order 'vert1 < vert2' broken
	// and some pairs to be invalid or duplicated
	changePairsOneVert(pairs2, vert2, vert1);

	// Sort in case of non-duplicated merge
	// When sorting the pair_arr, the array is sorted based on the two vertices
	// indices (you may refer to the definition of class 'CollapsablePair' and
	// function 'pair_comp'). So there may result in two types of equals (equal
	// elements reside continuously in the sorted array), one is the value equal,
	// which means that they point to different structs which have the same value
	// in vert1, vert2 field, one is pointer equal which means the two different
	// pointer points to the same struct. So these may need some special treatment
	// when merging
	sort(pairs1.pointer(0), pairs1.pointer(pairs1.count()), pair_comp);
	sort(pairs2.pointer(0), pairs2.pointer(pairs2.count()), pair_comp);

	// trim the NULL pointers in the end
	for (i = pairs1.count() - 1; i >= 0 && pairs1[i] == NULL; i --) ;
	pairs1.setCount(i + 1);
	for (i = pairs2.count() - 1; i >= 0 && pairs2[i] == NULL; i --) ;
	pairs2.setCount(i + 1);

	pair_arr new_pairs;
	new_pairs.clear();
	new_pairs.resize(pairs1.count() + pairs2.count());

	/* merge */
	for (i = 0, j = 0; i < pairs1.count() || j < pairs2.count();) {

		// pairs[i] and pairs[j] points to the same struct
		if (i < pairs1.count() && j < pairs2.count() && pairs1[i] == pairs2[j]) {
			
			///updateValidPairOrRemove(pairs1[i], new_pairs);

			// this is the collapsed pair
			pair_heap.remove(pairs1[i]);
			delete[] pairs1[i];
			i ++; j ++;
		}
		// pairs[i] and pairs[j] points to different value equal structs
		else if (i < pairs1.count() && j < pairs2.count() && *pairs1[i] == *pairs2[j]) {
			
			///updateValidPairOrRemove(pairs1[i], new_pairs);

			// these are the duplicated pairs
			// update arbitrary one and remove another 
			new_pairs.push_back(pairs1[i]);
			pair_heap.remove(pairs2[j]);

			// the pairs2[j] refers to another vertex which also maintains
			// a pointer to the same structure pairs2[j] points to, when
			// decimating the structure pairs2[j] points to, set the pointer
			// of the corresponding vertex to NULL
			setOnePairNull(pairs2[j]->getAnotherVert(vert1), pairs2[j]);
			delete[] pairs2[j];

			i ++; j ++;
		}
		else if (j >= pairs2.count() || i < pairs1.count() && *pairs1[i] < *pairs2[j]) {

			///updateValidPairOrRemove(pairs1[i], new_pairs);

			new_pairs.push_back(pairs1[i]);
			i ++;
		}
		else {
			///updateValidPairOrRemove(pairs2[j], new_pairs);

			new_pairs.push_back(pairs2[j]);
			j ++;
		}
	}

	/* post process */
	// the variable 'pair' is invalid now!!
	reevaluatePairs(new_pairs);
	pairs1.swap(new_pairs);
	pairs2.freeSpace();
}

void PairCollapse::reevaluatePairs(pair_arr &pairs) {

	for (int i = 0; i < pairs.count(); i ++) {
		evaluatePair(pairs[i]);
		pair_heap.update(pairs[i]);
	}
}

void PairCollapse::changeFacesOneVert(face_arr &face_indices, uint orig, uint dst) {
	
	for (int i = 0; i < face_indices.count(); i ++) 
		if (faces[face_indices[i]].valid()) {

			faces[face_indices[i]].changeOneVert(orig, dst);
			if (!faces[face_indices[i]].indexValid()) 
				valid_faces --;
		}
}

void PairCollapse::mergeFaces(uint vert1, uint vert2) {

	int i, j;

	/* pre process */

	face_arr &faces1 = vertices[vert1].adjacent_faces;
	face_arr &faces2 = vertices[vert2].adjacent_faces;
	// change the index of vert2 to vert1 for all faces adjacent
	// to vert2, this may cause some faces to be invalid or duplicated
	changeFacesOneVert(faces2, vert2, vert1);

	sort(faces1.pointer(0), faces1.pointer(faces1.count()), faceIndexComp);
	sort(faces2.pointer(0), faces2.pointer(faces2.count()), faceIndexComp);

	face_arr new_faces;
	new_faces.clear();
	new_faces.resize(faces1.count() + faces2.count());

	/* merge */
	for (i = 0, j = 0; i < faces1.count() || j < faces2.count(); ) {
		
		// the same face
		if (i < faces1.count() && j < faces2.count() && faces1[i] == faces2[j]) {
			if (faces[faces1[i]].valid()) 
				new_faces.push_back(faces1[i]);
			//else
			//	valid_faces --;
			i ++; j ++;
		}
		// two faces equal after the collapse
		else if (i < faces1.count() && j < faces2.count() && faces[faces1[i]] == faces[faces2[j]]) {
			if (faces[faces1[i]].valid()) {
				new_faces.push_back(faces1[i]);
				faces[faces2[j]].invalidate();
				valid_faces --;
			}
			//else
			//	valid_faces -= 2;
			i ++; j ++;
		}
		else if (j >= faces2.count() || i < faces1.count() && faces[faces1[i]] < faces[faces2[j]]) {
			if (faces[faces1[i]].valid()) 
				new_faces.push_back(faces1[i]);
			//else 
			//	valid_faces --;
			i ++;
		}
		else {
			if (faces[faces2[j]].valid()) 
				new_faces.push_back(faces2[j]);
			//else 
			//	valid_faces --;
			j ++;
		}
	}

	/* post process */
	faces1.swap(new_faces);
	faces2.freeSpace();
}

void PairCollapse::collectEdgeFaces(uint vert1, uint vert2, face_arr &_faces) {

	face_arr &faces1 = vertices[vert1].adjacent_faces;
	face_arr &faces2 = vertices[vert2].adjacent_faces;

	markFaces(faces1, 0);
	markFaces(faces2, 1);
	collectMarkFaces(faces1, _faces, 1);
}

void PairCollapse::markFaces(face_arr &_faces, unsigned char m) {
	
	for (int i = 0; i < _faces.count(); i ++)
		faces[_faces[i]].markFace(m);
}

void PairCollapse::collectMarkFaces(face_arr &faces_in, face_arr &faces_out, unsigned char m) {

	faces_out.clear();
	faces_out.resize(faces_in.count() / 2);

	for (int i = 0; i < faces_in.count(); i ++) 
		if (faces[faces_in[i]].markIs(m))
			faces_out.push_back(faces_in[i]);
}

#endif //__H_ITERATIVE_PAIR_COLLAPSE__ 