/*
 *  Iteratively perform the vertex pair collapse
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */

#ifndef __H_ITERATIVE_PAIR_COLLAPSE__
#define __H_ITERATIVE_PAIR_COLLAPSE__ 

// for MxBlock
#define HAVE_CASTING_LIMITS

#define ARRAY_NORMAL	0
#define ARRAY_USE_HASH	1

//#define _VERBOSE

#define ARRAY_USE	ARRAY_USE_HASH

#ifdef ARRAY_USE
	#if ARRAY_USE < 0 || ARRAY_USE > 1
		#error invalid value for ARRAY_USE macro 	
	#endif
#else
	#define ARRAY_USE	ARRAY_NORMAL
#endif

#include <algorithm>
#include <string.h>
#include <iostream>
#include <fstream>
#include <boost/unordered_map.hpp>
#include <libs/unordered/examples/fnv1.hpp>
#include "pcol_vertex.h"
#include "pcol_other_structures.h"
#include "common_types.h"
#include "h_dynamarray.h"
#include "MixKit/MxHeap.h"
#include "h_aug_time.h"

using std::ofstream;
using std::cout;
using std::endl;
using std::ostringstream;

inline std::size_t hash_value(uint i) {
	ostringstream oss;
	oss << i;
	hash::fnv_1a fnv;
	return fnv(oss.str());
}

using boost::unordered::unordered_map;
typedef unordered_map<uint, CollapsableVertex> ECVertexMap;
typedef unordered_map<uint, CollapsableFace> ECFaceMap;

typedef CollapsablePair* pCollapsablePair;

#if ARRAY_USE == ARRAY_NORMAL
	#define _for_loop(container, container_type) for (int __index = 0; __index < (container).count(); __index ++)
	#define _retrieve_elem(container) (container)[__index]
	#define _retrieve_index() __index
#else
	#define _for_loop(container, container_type) for (container_type::iterator __iter = (container).begin(); __iter != (container).end(); __iter ++)
	#define _retrieve_elem(container) __iter->second
	#define _retrieve_index() __iter->first
#endif

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

	#if ARRAY_USE == ARRAY_NORMAL
		return faces->at(face_index1).unsequencedLessThan(faces->at(face_index2));
	#else
		ECFaceMap::iterator iter1 = faces->find(face_index1), iter2 = faces->find(face_index2);
		return (iter1->second).unsequencedLessThan(iter2->second);
	#endif
	}

#if ARRAY_USE == ARRAY_NORMAL
	void setFaces(HDynamArray<CollapsableFace> *_faces) { faces = _faces; }

private:
	// thread safe ??
	HDynamArray<CollapsableFace> *faces;
#else
	void setFaces(ECFaceMap *_faces) { faces = _faces; }

private:
	// thread safe ??
	ECFaceMap *faces;
#endif
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
	virtual void initialize();
	void initValids();
	inline void unreferVertsCheck();
	
	
	////////////////////////////////////
	// Iterating
	
	// simplify targeting vertex
	// this function should be overrided
	// in specific derivative class
	bool targetVert(uint target_count);
	// simplify targeting face
	// this function should be overrided
	// in specific derivative class
	bool targetFace(uint target_count);


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
	virtual void collapsePair(pCollapsablePair pair);

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
	inline void removeFace(uint i);

	///////////////////////////////////////
	// Accessors

	inline uint vertexCount() const;
	inline uint faceCount() const;
	uint validVerts() const { return valid_verts; }
	uint validFaces() const { return valid_faces; }
	// these accessors will not add any element if no such
	// element exists, otherwise it will return a random
	// new created element which are usually invalid one
	// not residing in the container, modifying this object
	// is meaningless
	inline CollapsableVertex& v(uint i);
	inline CollapsableFace& f(uint i);
	inline bool f_interior (int i);
	// !! test if a face is valid should always use this
	inline bool face_is_valid(uint i) const;


	///////////////////////////////////////
	// File I/O & Output Generation

	bool readPly(char* filename);
	bool writePly(char* filename);
	void generateOutputId();
	// for debug
	void outputIds(char* filename);


	///////////////////////////////////////
	// Other Than Simplification

	void addInfo(std::string s);
	std::string getInfo() { return info; };
	void clearInfo() { info = ""; };
	void totalTime();

	// clear heap
	void clear();

	inline void facesToStr(face_arr &faces, string &str);

protected:
	/////////////////////////////////////
	// constants
	static const uint	DFLT_STAR_FACES = 6;
	static const uint	DFLT_STAR_PAIRS = 6;
	static const uint	INFO_BUF_CAPACITY = 1000;

#if ARRAY_USE == ARRAY_NORMAL
	HDynamArray<CollapsableVertex>	vertices;
	HDynamArray<CollapsableFace>	faces;
#else
	ECVertexMap	vertices;
	ECFaceMap	faces;
#endif
	uint	valid_verts;
	uint	valid_faces;
	MxHeap	pair_heap;

	FaceIndexComp	faceIndexComp;

	string info;

	HAugTime read_time, run_time, write_time;

	/////////////////////////////////////
	// assisting temporal variables
	CollapsableVertex	cvert;
	CollapsableFace	cface;
	vert_arr	starVerts1, starVerts2;

#ifdef _VERBOSE
	int merge_face_count;
	int last_valid_faces;
	int another_valid_faces;
	ofstream fverbose;
#endif
};

void PairCollapse::unreferVertsCheck() {
	valid_verts = 0;

#if ARRAY_USE == ARRAY_NORMAL
	for (int i = 0; i < vertices.count(); i ++) 
		if (!v(i).unreferred()) 
			valid_verts ++;

#elif ARRAY_USE == ARRAY_USE_HASH
	for (ECVertexMap::iterator iter = vertices.begin(); iter != vertices.end(); ) {
		CollapsableVertex& cvert = iter->second;
		if (!cvert.unreferred()) {
			valid_verts ++;
			iter ++;
		}
		else
			iter = vertices.erase(iter);
	}
#endif
}

void PairCollapse::collectStarVertices(uint vert_index, vert_arr *starVertices) {
	starVertices->clear();
	CollapsableVertex& cvert = v(vert_index);

	for (int i = 0; i < cvert.adjacent_faces.count(); i ++) {
		CollapsableFace& cface = f(cvert.adjacent_faces[i]);
		
		if (cface.i != vert_index && !starVertices->exist(cface.i))
			starVertices->push_back(cface.i);
		if (cface.j != vert_index && !starVertices->exist(cface.j))
			starVertices->push_back(cface.j);
		if (cface.k != vert_index && !starVertices->exist(cface.k))
			starVertices->push_back(cface.k);
	}
}

void PairCollapse::addCollapsablePair(CollapsablePair *new_pair) {
	v(new_pair->vert1).adjacent_col_pairs.push_back(new_pair);
	v(new_pair->vert2).adjacent_col_pairs.push_back(new_pair);

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
	} else {
		pair_heap.remove(pair);
		delete[] pair;
	}
}

void PairCollapse::setOnePairNull(uint vert, pCollapsablePair pair) {
	pair_arr &pairs = v(vert).adjacent_col_pairs;
	for	(int i = 0; i < pairs.count(); i ++)
		if (pairs[i] == pair) 
			pairs[i] = NULL;
}

void PairCollapse::mergePairs(uint vert1, uint vert2) {
	int i, j;

	/* pre process */
	pair_arr &pairs1 = v(vert1).adjacent_col_pairs;
	pair_arr &pairs2 = v(vert2).adjacent_col_pairs;
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
	/* the variable 'pair' is invalid now!! */
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
	for (int i = 0; i < face_indices.count(); i ++) {
		CollapsableFace &cface = f(face_indices[i]);
		if (cface.valid())  {
			cface.changeOneVert(orig, dst);
			//cface.sortIndex();
		#ifdef _VERBOSE
			if (!f(face_indices[i]).indexValid()) 
				another_valid_faces --;
		#endif
		}
	}
}

void PairCollapse::mergeFaces(uint vert1, uint vert2) {
	int i, j;

	/* pre process */
	face_arr &_faces1 = v(vert1).adjacent_faces;
	face_arr &_faces2 = v(vert2).adjacent_faces;

	face_arr faces1, faces2;
	faces1.resize(_faces1.count());
	faces2.resize(_faces2.count());

	for (int i = 0; i < _faces1.count(); i ++) {
		//if (_faces1[i] == 35148) {
		//	CollapsableFace cface = f(_faces1[i]);
		//}
		if (face_is_valid(_faces1[i])) 
			faces1.push_back(_faces1[i]);
	}

	for (int i = 0; i < _faces2.count(); i ++) {
		//if (_faces2[i] == 35148) {
		//	CollapsableFace cface = f(_faces2[i]);
		//}
		if (face_is_valid(_faces2[i])) 
			faces2.push_back(_faces2[i]);
	}

#ifdef _VERBOSE
	last_valid_faces = valid_faces;
	if (merge_face_count == 18) {
		int verbose = 0;
	}
	string str1;
	facesToStr(faces1, str1);
	string str2;
	facesToStr(faces2, str2);
#endif

	// change the index of vert2 to vert1 for all faces adjacent
	// to vert2, this may cause some faces to be invalid or duplicated
	changeFacesOneVert(faces2, vert2, vert1);

#ifdef _VERBOSE
	string str3;
	facesToStr(faces2, str3);
#endif

	sort(faces1.pointer(0), faces1.pointer(faces1.count()), faceIndexComp);
	sort(faces2.pointer(0), faces2.pointer(faces2.count()), faceIndexComp);

	face_arr new_faces;
	new_faces.clear();
	new_faces.resize(faces1.count() + faces2.count());

	/* merge */
	for (i = 0, j = 0; i < faces1.count() || j < faces2.count(); ) {
		// the same face
		if (i < faces1.count() && j < faces2.count() && faces1[i] == faces2[j]) {
			if (face_is_valid(faces1[i])) 
				new_faces.push_back(faces1[i]);
			else {
				removeFace(faces1[i]);
				//valid_faces --;
			}
			i ++; j ++;
		}
		// two faces equal after the collapse
		else if (i < faces1.count() && j < faces2.count() && f(faces1[i]).unsequencedEqual(f(faces2[j]))) {
			if (face_is_valid(faces1[i])) {
				new_faces.push_back(faces1[i]);
				f(faces2[j]).invalidate();
				removeFace(faces2[j]);
				//valid_faces --;
			#ifdef _VERBOSE
				another_valid_faces --;
			#endif
			} else {
				removeFace(faces1[i]);
				removeFace(faces2[j]);
				//valid_faces -= 2;
			}
			i ++; j ++;
		}
		else if (j >= faces2.count() || i < faces1.count() && f(faces1[i]).unsequencedLessThan(f(faces2[j]))) {
			if (face_is_valid(faces1[i])) 
				new_faces.push_back(faces1[i]);
			else {
				removeFace(faces1[i]);
				//valid_faces --;
			}
			i ++;
		}
		else {
			if (face_is_valid(faces2[j])) 
				new_faces.push_back(faces2[j]);
			else {
				removeFace(faces2[j]);
				//valid_faces --;
			}
			j ++;
		}
	}

	/* post process */
	_faces1.swap(new_faces);
	_faces2.freeSpace();

#ifdef _VERBOSE
	fverbose << "#" << merge_face_count ++ << " valid: " << valid_faces << " "  << last_valid_faces - valid_faces << "-" << endl;
	fverbose << "\tanother valid: " << another_valid_faces << endl;
#endif
}

void PairCollapse::collectEdgeFaces(uint vert1, uint vert2, face_arr &_faces) {
	face_arr &faces1 = v(vert1).adjacent_faces;
	face_arr &faces2 = v(vert2).adjacent_faces;

	markFaces(faces1, 0);
	markFaces(faces2, 1);
	collectMarkFaces(faces1, _faces, 1);
}

void PairCollapse::markFaces(face_arr &_faces, unsigned char m) {
	for (int i = 0; i < _faces.count(); i ++)
		f(_faces[i]).markFace(m);
}

void PairCollapse::collectMarkFaces(face_arr &faces_in, face_arr &faces_out, unsigned char m) {
	faces_out.clear();
	faces_out.resize(faces_in.count() / 2);

	for (int i = 0; i < faces_in.count(); i ++) 
		if (f(faces_in[i]).markIs(m))
			faces_out.push_back(faces_in[i]);
}

uint PairCollapse::vertexCount() const { 
#if ARRAY_USE == ARRAY_NORMAL
	return vertices.count(); 
#else
	return vertices.size();
#endif
}
uint PairCollapse::faceCount() const {
#if ARRAY_USE == ARRAY_NORMAL
	return faces.count();
#else
	return faces.size();
#endif
}

CollapsableVertex& PairCollapse::v(uint i) { 
//#if ARRAY_USE == ARRAY_NORMAL
	return vertices.at(i);
//#else
//	ECVertexMap::iterator iter = vertices.find(i);
//	if (iter != vertices.end())
//		return iter->second;
//	else
//		return CollapsableVertex();
//#endif
}

CollapsableFace& PairCollapse::f(uint i) { 
//#if ARRAY_USE == ARRAY_NORMAL
	return faces.at(i);
//#else
//	ECFaceMap::iterator iter = faces.find(i);
//	if (iter != faces.end())
//		return iter->second;
//	else
//		return CollapsableFace(0, 0, 0); // return a invalid face
//#endif
}

bool PairCollapse::f_interior (int i) {
	return v(f(i).i).interior() && 
		v(f(i).j).interior() && v(f(i).k).interior(); 
}

void PairCollapse::removeFace(uint i) {
#if ARRAY_USE == ARRAY_USE_HASH
	//CollapsableFace cface = f(i);
	//uint n = faces.erase(i);
	faces.erase(i);
	//cface = f(i);
#endif
	valid_faces --;
}

bool PairCollapse::face_is_valid(uint i) const {
#if ARRAY_USE == ARRAY_NORMAL
	return f(i).valid();
#else
	ECFaceMap::const_iterator iter = faces.find(i);
	if (iter != faces.end())
		return iter->second.valid();
	else
		return false;
#endif
}

void PairCollapse::facesToStr(face_arr &ifaces, string &str) {
	ostringstream oss;
	for (int i = 0; i < ifaces.count(); i ++) {
		CollapsableFace cface = f(ifaces[i]);
		oss << "<" << cface.i << ", " << cface.j << ", " << cface.k << "> ";
	}
	str = oss.str();
}

#endif //__H_ITERATIVE_PAIR_COLLAPSE__ 