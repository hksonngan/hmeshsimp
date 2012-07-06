/*
 *  uitility & common classes for algorithms
 *
 *  author: ht
 *  email : waytofall916@gmail.com
 */

#ifndef __UTIL_COMMON__
#define __UTIL_COMMON__

#include "math/chapill_vec3.h"
#include "math/chapill_vec4.h"
#include "math/mat44.h"


/* -- types & constants -- */

/* integer index type */
typedef int Integer;
typedef unsigned int uint;

enum WhichSide
{ Side1, Side2 };


/* class defined */
template<class T> class HVec3;
class HFaceFormula;
template<class FloatType> class HQEMatrix;
class HSoupTriangle;
template<class ElemType> class HTripleIndex;
class HFaceIndex;

typedef ChapillVec3<float> HVertex;
typedef ChapillVec3<float> HNormal;

/* 3 vector */
template<class T>
class HVec3
{
public:
	HVec3() {}

	HVec3(T _x, T _y, T _z) {
		x = _x; y = _y; z = _z; }

	void set(T _x, T _y, T _z) {
		x = _x; y = _y; z = _z; }

	void set(const HVec3 v) {
		x = v.x; y = v.y; z = v.z; }

	HVec3& operator+= (const HVec3 v) {
		x += v.x; y += v.y; z += v.z; 
		return *this;
	}

	HVec3 operator* (T n) const {
		HVec3 v(x * n, y * n, z * n);		
		return v;
	}

	HVec3 operator+ (HVec3 &vec) const {
		HVec3 v(x + vec.x, y + vec.y, z + vec.z);
		return v;
	}

	bool operator== (const HVec3 &vec) const {
		return x == vec.x && y == vec.y && z == vec.z;
	}

	bool operator!= (const HVec3 &vec) const {
		return !operator==(vec);
	}

public:
	T x, y, z;
};

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
	bool unsequncedEqual(const HTripleIndex &trip) const {
		this->sortIndex(arr1);
		trip.sortIndex(arr2);

		return arr1[0] == arr2[0] && arr1[1] == arr2[1] && arr1[2] == arr2[2];
	}

	bool operator< (const HTripleIndex &trip_ind) const
	{
		if (this->i < trip_ind.i)
			return true;
		else if (this->j < trip_ind.j)
			return true;
		else if (this->k < trip_ind.k)
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
	static ElemType arr1[3], arr2[3], temp;
	static int index1, index2, index3;
};

template<class ElemType> ElemType HTripleIndex<ElemType>::arr1[3];
template<class ElemType> ElemType HTripleIndex<ElemType>::arr2[3];
template<class ElemType> ElemType HTripleIndex<ElemType>::temp;
template<class ElemType> int HTripleIndex<ElemType>::index1;
template<class ElemType> int HTripleIndex<ElemType>::index2;
template<class ElemType> int HTripleIndex<ElemType>::index3;

/* face index: three HTripleIndex as cluster index */
class HFaceIndex
{
public:
	HFaceIndex() {}

	HFaceIndex(HTripleIndex<Integer>& tr1, HTripleIndex<Integer>& tr2, HTripleIndex<Integer>& tr3) {
		this->v1CIndex = tr1; this->v2CIndex = tr2; this->v3CIndex = tr3;
	}

	void set(HTripleIndex<Integer>& tr1, HTripleIndex<Integer>& tr2, HTripleIndex<Integer>& tr3) {
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
	HTripleIndex<Integer> v1CIndex, v2CIndex, v3CIndex;
};

#endif //__UTIL_COMMON__
