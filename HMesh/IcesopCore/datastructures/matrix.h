/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 定义3*3、4*4的double类型方阵:
** 取自Gems7内核数据结构代码
** 可对矩阵进行赋值、加减、数乘除、矩阵乘除、求逆等操作
** 另外, 还定义了二维向量与3*3矩阵及三维向量与4*4矩阵的相乘操作
**
** Author : shan @2011
**
****************************************************************************/


///////////////////////////////////////////////////////////////////////////////////////
//	Revised by Peng Yu 2003-10-23

#ifndef	MATRIX_H
#define MATRIX_H

#include "point.h"
#include "../math/mathlib.h"

namespace icesop {

class Matrix3
{             
private:
	void Dummy ();

public:
    double mat[3][3];

public:
 	//构造函数
	Matrix3 ();
	Matrix3 (const Matrix3& m);

	//赋值操作符
	void operator += (const Matrix3& m);
	void operator -= (const Matrix3& m);
	void operator *= (const Matrix3& m);
	void operator *= (double num);
	void operator /= (double num);

	//矩阵加
	Matrix3 operator + (const Matrix3& m) const;
	//矩阵减
	Matrix3 operator - (const Matrix3& m) const;
	//矩阵乘
	Matrix3 operator * (const Matrix3& m) const;
	//数乘
	Matrix3 operator * (double num) const;
	//数除
	Matrix3 operator / (double num) const;
	//求逆矩阵
	Matrix3 operator ! () const;
	
	void SetZero ();
	void SetUnit ();
	void SetShift (double dx, double dy);
	void SetScale (double sx, double sy);
	void SetRotate (double angle);
	void SetTranslate (double dx, double dy);
	
};

class Matrix4
{
private:
	void Dummy ();

public:
	 double mat[4][4];

public:
	Matrix4 ();
	Matrix4 (const Matrix4& m);

	//赋值操作符
	void operator += (const Matrix4& m);
	void operator -= (const Matrix4& m);
	void operator *= (const Matrix4& m);
	void operator *= (double num);
	void operator /= (double num);

	//矩阵加
	Matrix4 operator + (const Matrix4& m) const;
	//矩阵减
	Matrix4 operator - (const Matrix4& m) const;
	//矩阵乘
	Matrix4 operator * (const Matrix4& m) const;
	//数乘
	Matrix4 operator * (double num) const;
	//数除
	Matrix4 operator / (double num) const;
	//求逆矩阵
	Matrix4 operator ! () const;

	void SetZero ();
	void SetUnit ();
	void SetShift (double dx, double dy, double dz);
	void SetScale (double sx, double sy, double sz);
	void SetRotate (int axis, double angle);
	void SetRotateV (const Vector3D& v, double ang);
	void SetRotateZ (const Vector3D& v);
	void SetLocal (const Point3D& p0, const Vector3D& zv, const Vector3D& xv);
	void SetMirror (double a, double b, double c, double d);
	void SetTranslate (double dx, double dy, double dz);
};

Point2D operator * (const Point2D& p, const Matrix3& mat);
Vector2D operator * (const Vector2D& p, const Matrix3& mat);

Point3D operator * (const Point3D& p, const Matrix4& mat);
Vector3D operator * (const Vector3D& p, const Matrix4& mat);

void glmat2mat(double glmat[16], Matrix4 &mat);
void mat2glmat(const Matrix4 &mat, double glmat[16]);

} // namespace icesop

#endif // MATRIX_H
