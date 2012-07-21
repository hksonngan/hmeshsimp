/*
 *  Math Classes
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __H_MATH__
#define __H_MATH__

#include "common_types.h"

enum WhichSide
{ Side1, Side2 };

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

#endif //__H_MATH__