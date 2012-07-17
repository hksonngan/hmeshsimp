/*
 *  Utility & common definitions for algorithms
 *
 *  Author: Ht
 *  Email : waytofall916@gmail.com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __UTIL_COMMON__
#define __UTIL_COMMON__

#include "math/chapill_vec3.h"
#include "math/chapill_vec4.h"
#include "math/mat44.h"
#include "gfx/vec3.h"


/* -- macros -- */

#define WRITE_BLOCK(out, n, size)			out.write((char *)&n, size)
#define WRITE_UINT(out, n)					out.write((char *)&n, sizeof(uint))

#define READ_BLOCK(in, n, size)				in.read((char *)&n, size)
#define READ_UINT(in, n)					in.read((char *)&n, sizeof(uint))

#define C_READ_BLOCK(fp, n, size, count)	fread((void *)&n, size, count, fp)
#define C_WRITE_BLOCK(fp, n, size, count)	fwrite((void *)&n, size, count, fp)

/* -- types & constants -- */

typedef int integer;
typedef unsigned int uint;

#define MAX_UINT	0xffffffff
#define INVALID_CLUSTER_INDEX	MAX_UINT

enum WhichSide
{ Side1, Side2 };

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

/* calculate face formula */
class HFaceFormula
{
public:
	static void calcTriangleFaceFormula(HVertex _v1, HVertex _v2, HVertex _v3);
	static float calcTriangleFaceArea(HVertex &_v1, HVertex &_v2, HVertex &_v3);
	static float calcD(HNormal nm, HVertex v);
	static WhichSide sideOfPlane(HNormal, float d, HVertex v);
	static WhichSide sideOfPlane(float a, float b, float c, float d, HVertex v) {
		return sideOfPlane(HNormal(a, b, c), d, v); }
	static void calcFaceNormal(HVertex v1, HVertex v2, HVertex v3, HNormal &n);

	void setStatic() {
		_a = a; _b = b; _c = c; _d = d; }
	
	void set(HNormal n, float __d) {
		_a = n.x; _b = n.y; _c = n.z; _d = __d;
	}

public:
	// parameters of a plane
	static float a, b, c, d;

	float _a, _b, _c, _d;
};

/* triangle in a triangle soup */
class HSoupTriangle
{
public:
	void set(HVertex _v1, HVertex _v2, HVertex _v3) {
		v1 = _v1; v2 = _v2; v3 = _v3; }

public:
	HVertex v1, v2, v3;
};

/* quadric error matrix */
template<class FloatType>
class HQEMatrix
{
public:
	HQEMatrix() { setZero(); }

	inline void setZero();

	HQEMatrix(HSoupTriangle tri) {
		calcQem(tri); }

	inline HQEMatrix<FloatType>& operator+= (const HQEMatrix<FloatType>& m);

	void calcQem(HSoupTriangle tri) {
		HFaceFormula::calcTriangleFaceFormula(tri.v1, tri.v2, tri.v3);
		calcQem(HFaceFormula::a, HFaceFormula::b, HFaceFormula::c, HFaceFormula::d);
	}

	void calQem(HVertex &v1, HVertex &v2, HVertex &v3) {
		HFaceFormula::calcTriangleFaceFormula(v1, v2, v3);
		calcQem(HFaceFormula::a, HFaceFormula::b, HFaceFormula::c, HFaceFormula::d);
	}

	inline FloatType getError(HVertex);

	inline HQEMatrix<FloatType> operator* (FloatType f) const;
	inline HQEMatrix<FloatType>& operator*= (FloatType f);

	inline void calcQem(FloatType a, FloatType b, FloatType c, FloatType d);
	inline bool calcRepresentativeVertex(HVertex& vertex);

public:
	FloatType a11, a12, a13, a14;
	FloatType      a22, a23, a24;
	FloatType           a33, a34;
	FloatType                a44;
};

template<class FloatType>
void HQEMatrix<FloatType>::setZero()
{
	a11 = a12 = a13 = a14 = 0;
	a22 = a23 = a24 = 0;
	a33 = a34 = 0;
	a44 = 0;
}

template<class FloatType>
HQEMatrix<FloatType>& HQEMatrix<FloatType>::operator+= (const HQEMatrix<FloatType>& m)
{
	a11 += m.a11; a12 += m.a12; a13 += m.a13; a14 += m.a14;
	a22 += m.a22; a23 += m.a23; a24 += m.a24;
	a33 += m.a33; a34 += m.a34;
	a44 += m.a44;

	return *this;
}

template<class FloatType>
void HQEMatrix<FloatType>::calcQem(FloatType a, FloatType b, FloatType c, FloatType d)
{
	a11 = a*a; a12 = a*b; a13 = a*c; a14 = a*d;
	a22 = b*b; a23 = b*c; a24 = b*d;
	a33 = c*c; a34 = c*d;
	a44 = d*d;
}

template<class FloatType>
bool HQEMatrix<FloatType>::calcRepresentativeVertex(HVertex& vertex)
{
	mymath::Mat44<FloatType> mat(
		a11, a12, a13, a14,
		a12, a22, a23, a24,
		a13, a23, a33, a34,
		0 ,  0 ,  0 ,  1 );
	mymath::Mat44<FloatType> inv;

	if (!mat.Inverse(inv, mymath::Mat44INV_TOLERANCE))
		return false;

	mymath::ChapillVec4<float> new_vertex = inv * mymath::ChapillVec4<float>(0, 0, 0, 1);

	vertex.Set(new_vertex.x, new_vertex.y, new_vertex.z);

	return true;
}

template<class FloatType>
HQEMatrix<FloatType> HQEMatrix<FloatType>::operator* (FloatType f) const
{
	HQEMatrix<FloatType> q;

	q.a11 *= f; q.a12 *= f; q.a13 *= f; q.a14 *= f;
	q.a22 *= f; q.a23 *= f; q.a24  *= f;
	q.a33 *= f; q.a34 *= f;
	q.a44 *= f;

	return q;
}

template<class FloatType>
HQEMatrix<FloatType>& HQEMatrix<FloatType>::operator*= (FloatType f)
{
	a11 *= f; a12 *= f; a13 *= f; a14 *= f;
	a22 *= f; a23 *= f; a24  *= f;
	a33 *= f; a34 *= f;
	a44 *= f;

	return *this;
}

/* triple index */
template<class ElemType>
class HTripleIndex
{
public:
	HTripleIndex() {}

	HTripleIndex(ElemType _i, ElemType _j, ElemType _k) {
		i = _i; j = _j; k = _k; }

	void set(ElemType _i, ElemType _j, ElemType _k) {
		i = _i; j = _j; k = _k; }

	bool operator!= (const HTripleIndex &trip_ind) const {
		return i != trip_ind.i || j != trip_ind.j || k != trip_ind.k;
	}

	bool operator== (const HTripleIndex &trip_ind) const {
		return i == trip_ind.i && j == trip_ind.j && k == trip_ind.k;
	}

	// equals without considering the sequence
	bool unsequencedEqual(const HTripleIndex &trip) const {

		this->sortIndex(arr1);
		trip.sortIndex(arr2);

		return arr1[0] == arr2[0] && arr1[1] == arr2[1] && arr1[2] == arr2[2];
	}

	bool operator< (const HTripleIndex &trip_ind) const
	{
		if (this->i < trip_ind.i)
			return true;
		else if (this->i < trip_ind.i)
			return false;
		else if (this->j < trip_ind.j)
			return true;
		else if (this->j > trip_ind.j)
			return false;
		else if (this->k < trip_ind.k)
			return true;

		return false;
	}

	bool unsequencedLessThan(const HTripleIndex &trip_ind) const {
		
		this->sortIndex(arr1);
		trip_ind.sortIndex(arr2);
		
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
	void sortIndex(ElemType *arr) const
	{
		arr[0] = i;
		arr[1] = j;
		arr[2] = k;

		// insertion sort
		for (index1 = 1; index1 < 3; index1 ++) {

			for (index2 = 0; index2 < index1; index2 ++)
				if (arr[index2] > arr[index1])
					break;

			if (index1 == index2)
				continue;

			temp = arr[index1];
			for (index3 = index1; index3 > index2; index3 --) {
				arr[index3] = arr[index3 - 1];
			}
			arr[index2] = temp;
		}
	}

public:
	ElemType i, j, k;

private:
	// assisting variable for sorting and comparing
	static ElemType arr1[3], arr2[3], temp;
	static int index1, index2, index3;
};

template<class ElemType> ElemType HTripleIndex<ElemType>::arr1[3];
template<class ElemType> ElemType HTripleIndex<ElemType>::arr2[3];
template<class ElemType> ElemType HTripleIndex<ElemType>::temp;
template<class ElemType> int HTripleIndex<ElemType>::index1;
template<class ElemType> int HTripleIndex<ElemType>::index2;
template<class ElemType> int HTripleIndex<ElemType>::index3;

typedef HTripleIndex<uint> HFace;
typedef HTripleIndex<uint> HClusterIndex;

/* face index: three HTripleIndex as cluster index */
class HFaceIndex
{
public:
	HFaceIndex() {}

	HFaceIndex(HTripleIndex<uint>& tr1, HTripleIndex<uint>& tr2, HTripleIndex<uint>& tr3) {
		this->v1CIndex = tr1; this->v2CIndex = tr2; this->v3CIndex = tr3;
	}

	void set(HTripleIndex<uint>& tr1, HTripleIndex<uint>& tr2, HTripleIndex<uint>& tr3) {
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
	HTripleIndex<uint> v1CIndex, v2CIndex, v3CIndex;
};

class LRUVertex {
public:
	// binary(!) read
	bool read(ifstream& fin) {

		READ_BLOCK(fin, v.x, VERT_ITEM_SIZE);
		READ_BLOCK(fin, v.y, VERT_ITEM_SIZE);
		READ_BLOCK(fin, v.z, VERT_ITEM_SIZE);

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

		WRITE_BLOCK(fout, v.x, VERT_ITEM_SIZE);
		WRITE_BLOCK(fout, v.y, VERT_ITEM_SIZE);
		WRITE_BLOCK(fout, v.z, VERT_ITEM_SIZE);

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

private:
	static const uint VERT_ITEM_SIZE = sizeof(float);
};
#endif //__UTIL_COMMON__
