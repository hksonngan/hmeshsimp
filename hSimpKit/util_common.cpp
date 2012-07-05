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

float HFaceFormula::calcTriangleFaceArea(HVertex &_v1, HVertex &_v2, HVertex &_v3)
{
	Vec3<float> v1(_v1.x, _v1.y, _v1.z), 
		v2(_v2.x, _v2.y, _v2.z), 
		v3(_v3.x, _v3.y, _v3.z);

	Vec3<float> edge1(v1 - v2), 
		edge2(v2 - v3);

	Vec3<float> normal = edge1 ^ edge2; // cross product

	return normal.Length();
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