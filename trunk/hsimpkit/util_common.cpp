#include "util_common.h"

/* --- HFaceFormula --- */

float HFaceFormula::a = 0;
float HFaceFormula::b = 0;
float HFaceFormula::c = 0;
float HFaceFormula::d = 0;

void HFaceFormula::calcTriangleFaceFormula(HVertex _v1, HVertex _v2, HVertex _v3)
{
	HNormal edge1(_v1 - _v2), 
		edge2(_v2 - _v3);

	HNormal normal = edge1 ^ edge2; // cross product

	normal.Normalize();

	a = normal.x;
	b = normal.y;
	c = normal.z;

	d = - (a * _v1.x + b * _v1.y + c * _v1.z);
}

void HFaceFormula::calcFaceNormal(HVertex v1, HVertex v2, HVertex v3, HNormal &n) {

	HNormal edge1(v1 - v2), 
		edge2(v2 - v3);

	n = edge1 ^ edge2; // cross product
}

float HFaceFormula::calcTriangleFaceArea(HVertex &_v1, HVertex &_v2, HVertex &_v3)
{
	ChapillVec3<float> v1(_v1.x, _v1.y, _v1.z), 
		v2(_v2.x, _v2.y, _v2.z), 
		v3(_v3.x, _v3.y, _v3.z);

	ChapillVec3<float> edge1(v1 - v2), 
		edge2(v2 - v3);

	ChapillVec3<float> normal = edge1 ^ edge2; // cross product

	return normal.Length() / 2;
}

float HFaceFormula::calcD(HNormal nm, HVertex v)
{
	return - nm * v;
}

WhichSide HFaceFormula::sideOfPlane(HNormal nm, float d, HVertex v)
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
