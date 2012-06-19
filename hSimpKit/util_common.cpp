#include "util_common.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "math/mat44.h"


/* --- HFaceFormula --- */

float HFaceFormula::a = 0;
float HFaceFormula::b = 0;
float HFaceFormula::c = 0;
float HFaceFormula::d = 0;

/* calculate face formula */
void HFaceFormula::calcTriangleFaceFormula(HVertex _v1, HVertex _v2, HVertex _v3)
{
	Vec3<float> v1(_v1.x, _v1.y, _v1.z), 
		v2(_v2.x, _v2.y, _v2.z), 
		v3(_v3.x, _v3.y, _v3.z);

	Vec3<float> edge1(v1 - v2), 
		edge2(v2 - v3);

	Vec3<float> normal = edge1 ^ edge2; // cross product

	normal.Normalize();

	a = normal.x;
	b = normal.y;
	c = normal.z;

	d = - (a * v1.x + b * v1.y + c * v1.z);
}

float HFaceFormula::calcTriangleFaceArea(HVertex _v1, HVertex _v2, HVertex _v3)
{
	Vec3<float> v1(_v1.x, _v1.y, _v1.z), 
		v2(_v2.x, _v2.y, _v2.z), 
		v3(_v3.x, _v3.y, _v3.z);

	Vec3<float> edge1(v1 - v2), 
		edge2(v2 - v3);

	Vec3<float> normal = edge1 ^ edge2; // cross product

	return normal.Length();
}

inline float HFaceFormula::calcD(HNormal nm, HVertex v)
{
	return - nm * v;
}

inline WhichSide HFaceFormula::sideOfPlane(HNormal nm, float d, HVertex v)
{
	if (nm.x < 0) {
		nm = - nm;
		d = - d;
	}

	float r = nm * v + d;

	if (r > 0)
		return Side1;
	else
		return Side2;
}

/* --- HQEMatrix --- */

void HQEMatrix::setZero()
{
	a11 = a12 = a13 = a14 = 0;
	a22 = a23 = a24 = 0;
	a33 = a34 = 0;
	a44 = 0;
}

HQEMatrix& HQEMatrix::operator+= (const HQEMatrix& m)
{
	a11 += m.a11; a12 += m.a12; a13 += m.a13; a14 += m.a14;
	a22 += m.a22; a23 += m.a23; a24 += m.a24;
	a33 += m.a33; a34 += m.a34;
	a44 += m.a44;

	return *this;
}

void HQEMatrix::calcQem(float a, float b, float c, float d)
{
	a11 = a*a; a12 = a*b; a13 = a*c; a14 = a*d;
	a22 = b*b; a23 = b*c; a24 = b*d;
	a33 = c*c; a34 = c*d;
	a44 = d*d;
}

bool HQEMatrix::calcRepresentativeVertex(HVertex& vertex)
{
	mymath::Mat44<float> mat(
		a11, a12, a13, a14,
		a12, a22, a23, a24,
		a13, a23, a33, a34,
		0 ,  0 ,  0 ,  1 );
	mymath::Mat44<float> inv;

	if (!mat.Inverse(inv, mymath::Mat44INV_TOLERANCE))
		return false;

	mymath::Vec4<float> new_vertex = inv * mymath::Vec4<float>(0, 0, 0, 1);

	vertex.Set(new_vertex.x, new_vertex.y, new_vertex.z);

	return true;
}

HQEMatrix HQEMatrix::operator* (float f) const
{
	HQEMatrix q;

	q.a11 *= f; q.a12 *= f; q.a13 *= f; q.a14 *= f;
	q.a22 *= f; q.a23 *= f; q.a24  *= f;
	q.a33 *= f; q.a34 *= f;
	q.a44 *= f;

	return q;
}

HQEMatrix& HQEMatrix::operator*= (float f)
{
	a11 *= f; a12 *= f; a13 *= f; a14 *= f;
	a22 *= f; a23 *= f; a24  *= f;
	a33 *= f; a34 *= f;
	a44 *= f;

	return *this;
}