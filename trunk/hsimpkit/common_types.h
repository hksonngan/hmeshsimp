/*
 *  Common Types
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __UTIL_COMMON__
#define __UTIL_COMMON__

#include "math/chapill_vec3.h"
#include "math/chapill_vec4.h"
#include "math/mat44.h"
#include "gfx/vec3.h"
#include "lru_cache.h"
#include "common_def.h"
#include "io_common.h"
#include "h_algorithm.h"


typedef ChapillVec3<float> HVector3f;
typedef HVector3f HVertex;
typedef HVector3f HNormal;

template<class T1, class T2>
inline void assign(TVec3<T1> &v1, const ChapillVec3<T2> &v2) {

	v1[0] = v2.x;
	v1[1] = v2.y;
	v1[2] = v2.z;
}

template<class T1, class T2>
inline void assign(ChapillVec3<T1> &v1, const TVec3<T2> &v2) {

	v1.x = v2[0];
	v1.y = v2[1];
	v1.z = v2[2];
}

/* triangle in a triangle soup */
class HSoupTriangle
{
public:
	void set(HVertex _v1, HVertex _v2, HVertex _v3) {
		v1 = _v1; v2 = _v2; v3 = _v3; }

public:
	HVertex v1, v2, v3;
};


/* triple index */
template<class ElemType>
class HTriple
{
public:
	HTriple() {}

	HTriple(ElemType _i, ElemType _j, ElemType _k) {
		i = _i; j = _j; k = _k; }

	void set(ElemType _i, ElemType _j, ElemType _k) {
		i = _i; j = _j; k = _k; }

	bool operator!= (const HTriple &trip_ind) const {
		return i != trip_ind.i || j != trip_ind.j || k != trip_ind.k;
	}

	bool operator== (const HTriple &trip_ind) const {
		return i == trip_ind.i && j == trip_ind.j && k == trip_ind.k;
	}

	// equals without considering the sequence
	bool unsequencedEqual(const HTriple &trip) const {
		ElemType arr1[3], arr2[3];

		this->_sortIndex(arr1);
		trip._sortIndex(arr2);

		return arr1[0] == arr2[0] && arr1[1] == arr2[1] && arr1[2] == arr2[2];
	}

	bool operator< (const HTriple &trip_ind) const {
		if (this->i < trip_ind.i)
			return true;
		else if (this->i > trip_ind.i)
			return false;
		else if (this->j < trip_ind.j)
			return true;
		else if (this->j > trip_ind.j)
			return false;
		else if (this->k < trip_ind.k)
			return true;

		return false;
	}

	bool unsequencedLessThan(const HTriple &trip_ind) const {
		ElemType arr1[3], arr2[3];
		
		this->_sortIndex(arr1);
		trip_ind._sortIndex(arr2);
		
		if (arr1[0] < arr2[0])
			return true;
		else if (arr1[0] > arr2[0])
			return false;
		else if (arr1[1] < arr2[1])
			return true;
		else if (arr1[1] > arr2[1])
			return false;
		else if (arr1[2] < arr2[2])
			return true;

		return false;
	}

	// sort the index with ascending order so that 
	// it can be identified with the same indices 
	// occurring in different order
	void _sortIndex(ElemType *arr) const {
		arr[0] = i; arr[1] = j; arr[2] = k;
		// insertion sort
		insertion_sort<ElemType, ElemType*>(arr, 3);
	}

	void sortIndex() {
		ElemType arr[3];
		typedef ElemType ElemTypeArr3[3];
		arr[0] = i; arr[1] = j; arr[2] = k;
		insertion_sort<ElemType, ElemTypeArr3>(arr, 3);
		i = arr[0]; j = arr[1]; k = arr[2];
	}

public:
	ElemType i, j, k;
};

typedef HTriple<uint> HFace;
typedef HTriple<uint> HClusterIndex;

inline void write_face_txt(ostream &out, const HTriple<uint> &f) {

	out << "3 " << f.i << " " << f.j << " " << f.k << endl;
}

inline void write_vert_txt(ostream &out, const HVertex &v) {

	out << v.x << " " << v.y << " " << v.z << endl;
}

inline void write_face(ostream &out, const HTriple<uint> &f) {
	uchar n = 3;

	WRITE_BLOCK(out, n, sizeof(uchar));
	WRITE_UINT(out, f.i);
	WRITE_UINT(out, f.j);
	WRITE_UINT(out, f.k);
#ifndef WRITE_PATCH_BINARY
	out << endl;
#endif
}

inline void write_vert(ostream &out, const HVertex &v) {

	WRITE_BLOCK(out, v.x, VERT_ITEM_SIZE);
	WRITE_BLOCK(out, v.y, VERT_ITEM_SIZE);
	WRITE_BLOCK(out, v.z, VERT_ITEM_SIZE);
#ifndef WRITE_PATCH_BINARY
	out << endl;
#endif
}

inline bool face_comp(const HTriple<uint> &f1, const HTriple<uint> &f2) {

	return f1.unsequencedLessThan(f2);
}

/* face index: three HTriple as cluster index */
class HFaceIndex
{
public:
	HFaceIndex() {}

	HFaceIndex(HTriple<uint>& tr1, HTriple<uint>& tr2, HTriple<uint>& tr3) {
		this->v1CIndex = tr1; this->v2CIndex = tr2; this->v3CIndex = tr3;
	}

	void set(HTriple<uint>& tr1, HTriple<uint>& tr2, HTriple<uint>& tr3) {
		this->v1CIndex = tr1; this->v2CIndex = tr2; this->v3CIndex = tr3;
	}

	bool operator== (const HFaceIndex &index) const {
		return this->v1CIndex == index.v1CIndex && this->v2CIndex == index.v2CIndex 
			&& this->v3CIndex == index.v3CIndex;
	}

	bool operator!= (const HFaceIndex &index) const {
		return !operator==(index); }

	// used for hash compare functor
	bool operator< (const HFaceIndex &index) const
	{
		if (this->v1CIndex < index.v1CIndex)
			return true;
		else if (this->v2CIndex < index.v2CIndex)
			return true;
		else if (this->v3CIndex < index.v3CIndex)
			return true;

		return false;
	}

public:
	HTriple<uint> v1CIndex, v2CIndex, v3CIndex;
};

class LRUVertex {
public:
	// binary(!) read
	bool read(ifstream& fin) {

		READ_BLOCK_BIN(fin, v.x, VERT_ITEM_SIZE);
		READ_BLOCK_BIN(fin, v.y, VERT_ITEM_SIZE);
		READ_BLOCK_BIN(fin, v.z, VERT_ITEM_SIZE);

		if (fin.good())
			return true;
		return false; 
	}

	bool read(FILE *fp) { 

		if (C_READ_BLOCK(fp, v.x, VERT_ITEM_SIZE, 1) != 1)
			return false;
		if (C_READ_BLOCK(fp, v.x, VERT_ITEM_SIZE, 1) != 1)
			return false;
		if (C_READ_BLOCK(fp, v.x, VERT_ITEM_SIZE, 1) != 1)
			return false;

		return true; 
	}

	// binary(!) write
	bool write(ofstream& fout) { 

		WRITE_BLOCK_BIN(fout, v.x, VERT_ITEM_SIZE);
		WRITE_BLOCK_BIN(fout, v.y, VERT_ITEM_SIZE);
		WRITE_BLOCK_BIN(fout, v.z, VERT_ITEM_SIZE);

		if (fout.good())
			return true;
		return false; 
	}

	// hash the index
	static unsigned int hash(unsigned int index) { return index; }
	
	// the size of the 
	static size_t size() { return sizeof(HVertex); }

public:
	HVertex v;
};

typedef LRUCache<LRUVertex> VertexBinary;

#endif //__UTIL_COMMON__
