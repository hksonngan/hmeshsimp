/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 定义了二维、三维、四维矢量和点:
** 取自Gems7内核数据结构代码
** 可对矢量和点进行赋值、加减、点乘除、叉乘、等操作
** 另外, 还可对矢量进行单位化、取长度等操作
**
** Author : shan @2011
**
****************************************************************************/

#include <cassert>
#include <iostream>

namespace icesop {

//Vector2D
/*-------------------------------------------------------*/
Vector2D::Vector2D (double newx, double newy)
:x(newx), y(newy)
{
}

Vector2D& Vector2D::operator += (const Vector2D& v)
{ 
	x += v.x;		y += v.y; 
	return *this; 
}

Vector2D& Vector2D::operator -= (const Vector2D& v)
{ 
	x -= v.x;		y -= v.y; 
	return *this; 
}

Vector2D& Vector2D::operator *= (double num)
{
	x *= num;		y *= num; 
	return *this; 
}

Vector2D& Vector2D::operator /= (double num)
{ //ASSERT (fabs (num) < TOLER);
	x /= num;		y /= num; 
	return *this;
}
	
//单目减
Vector2D Vector2D::operator - () const
{ 
	return Vector2D (-x, -y); 
}

Vector2D::operator Point2D () const
{
	return Point2D (x, y); 
}

Vector3D Vector2D::ConvertTo3D() const
{
	return Vector3D(x, y, 0);   
}

Vector2D Vector2D::perpVector() const
{
	return Vector2D(-y,x);
}

//取元素
double& Vector2D::operator [] (unsigned int i)
{
	assert(i<2);
	double* p=reinterpret_cast<double*>(this);
	return p[i];
}

double Vector2D::operator [] (unsigned int i) const
{
	assert(i<2);
	const double* p=reinterpret_cast<const double*>(this);
	return p[i];
}

// 取长度
double Vector2D::Length ()  const                             
{ 
	return sqrt (x*x + y*y); 
}

void Vector2D::SetUnit ()
{
	double a = Length(); // * (1 + 1e-15);//by peng yu 2003-11-21
	if (a > TOLER) (*this) /= a;
}

//bi-op
double operator ^ (const Vector2D& u, const Vector2D& v)
{
	return( u.x* v.y- u.y* v.x );
}

bool operator == (const Vector2D& u, const Vector2D& v)	// 仅比较方向，忽略长度
{
	return Length2(u - v) <= TOLER * TOLER;	//  [3/31/2007 YangLei]
}

bool operator != (const Vector2D& u, const Vector2D& v)
{
	return !(u==v);
}

//矢量加
Vector2D operator + (const Vector2D& u, const Vector2D& v) 
{ 
	return Vector2D (u.x + v.x, u.y + v.y); 
}
//矢量减
Vector2D operator - (const Vector2D& u, const Vector2D& v) 
{ 
	return Vector2D (u.x - v.x, u.y - v.y); 
}
//矢量乘(点积)
double  operator * (const Vector2D& u, const Vector2D& v) 
{
	return u.x * v.x + u.y * v.y; 
}
//数乘
Vector2D operator * (const Vector2D& v, double num) 
{ 
	return Vector2D (v.x * num, v.y * num); 
}

Vector2D operator * (double num, const Vector2D& v)
{ 
	return Vector2D (v.x * num, v.y * num); 
}
//数除
Vector2D operator / (const Vector2D& v, double num) 
{ 
	//ASSERT (fabs (num) < TOLER);
	return Vector2D (v.x / num, v.y / num); 
}

//END Vector2D.
/*-------------------------------------------------------*/

//Vector3D.
/*-------------------------------------------------------*/
Vector3D::Vector3D (double newx, double newy, double newz)
:x(newx), y(newy), z(newz)
{
}

Vector3D& Vector3D::operator += (const Vector3D& v)
{ 
	x += v.x;	y += v.y;	z += v.z;  
	return *this; 
}

Vector3D& Vector3D::operator -= (const Vector3D& v)
{
	x -= v.x;	y -= v.y;	z -= v.z; 
	return *this; 
}

Vector3D& Vector3D::operator *= (double num)
{ 
	x *= num;	y *= num;	z *= num; 
	return *this; 
}

Vector3D& Vector3D::operator /= (double num)
{
	num = 1.0/num;	x *= num;	y *= num;	z *= num;
	//revised by LiuXiaoming 2007-6-12, 减少除法运算
	return *this;
}

Vector3D& Vector3D::operator ^= (const Vector3D& v)
{ 
	double a =   y * v.z - z * v.y;
	double b = - x * v.z + z * v.x;
	double c =   x * v.y - y * v.x;

	x = a;
	y = b;
	z = c;
	return *this;
}

Vector3D Vector3D::operator - () const
{ 
	return Vector3D (-x, -y, -z); 
}

Vector3D::operator Point3D () const
{
	return Point3D (x, y, z);
}

Vector3D Vector3D::perpVector() const
{
	Vector3D vecReturn;
	if(	fabs(x)<0.015625	&&	fabs(y)<0.015625 ) {
		vecReturn.x=z;
		vecReturn.y=0.0;
		vecReturn.z=-x;
	}
	else{
		vecReturn.x=-y;
		vecReturn.y=x;
		vecReturn.z=0.0;
	}

	return vecReturn;
}

//取元素
double& Vector3D::operator [] (unsigned int i)
{
	assert(i<3);
	double* p=reinterpret_cast<double*>(this);
	return p[i];
}

double Vector3D::operator[] (unsigned int i) const
{
	assert(i<3);
	const double* p=reinterpret_cast<const double*>(this);
	return p[i];
}

void Vector3D::setValue(double newx, double newy,double newz)
{
	x=newx;	y=newy;	z=newz;
}

//矢量加
Vector3D operator + (const Vector3D& u, const Vector3D& v) 
{ 
	return Vector3D (u.x + v.x, u.y + v.y, u.z + v.z);
}
//矢量减
Vector3D operator - (const Vector3D& u, const Vector3D& v) 
{ 
	return Vector3D (u.x - v.x, u.y - v.y, u.z - v.z);
}
//矢量乘(点积)
double operator * (const Vector3D& u, const Vector3D& v) 
{ 
	return (u.x * v.x+u.y * v.y+ u.z * v.z);
}
//矢量乘(叉积)
Vector3D operator ^ (const Vector3D& u, const Vector3D& v) 
{ 
	return Vector3D (u.y * v.z - u.z*v.y, 
					-u.x*v.z+u.z*v.x,
					u.x*v.y-u.y*v.x
				);
}
//数乘
Vector3D operator * (const Vector3D& v, double num)
{ 
	return Vector3D (v.x * num, v.y * num, v.z * num); 
}

Vector3D operator * (double num, const Vector3D& v) 
{ 
	return Vector3D (v.x * num, v.y * num, v.z*num); 
}
//数除
Vector3D operator / (const Vector3D& v, double num) 
{ 
	//ASSERT (fabs (num) < TOLER);
	num = 1.0/num;
	//return Vector3D (x / num, y / num, z / num); 
	return Vector3D (v.x * num, v.y * num, v.z * num);
	//revised by LiuXiaoming 2007-6-12, 减少除法运算
}

// 取长度
double Vector3D::Length ()  const                             
{ 
	return sqrt (x*x + y*y + z*z); 
}

void Vector3D::SetUnit ()
{
	double a = Length (); // * (1 + 1e-15);
	if (a > TOLER) (*this) /= a;
}

bool operator == (const Vector3D& u, const Vector3D& v) 
{
	Vector3D a (u), b (v);

	a.SetUnit ();
	b.SetUnit ();

	return Length2(a - b) <= TOLER * TOLER;
}

bool operator != (const Vector3D& u, const Vector3D& v)
{
	return !(u==v);
}


//END Vector3D
/*-------------------------------------------------------*/

//Point2D
/*-------------------------------------------------------*/
Point2D::Point2D (double newx, double newy)
:x(newx), y(newy)
{
}

Point2D& Point2D::operator += (const Vector2D& v)
{
	x += v.x;	y += v.y; 
	return *this;
}

Point2D& Point2D::operator -= (const Vector2D& v)
{ 
	x -= v.x; y -= v.y; 
	return *this;
}

Point2D Point2D::operator - () const
{
	return Point2D (-x, -y); 
}

Point3D Point2D::ConvertTo3D() const
{
	return Point3D(x, y, 0.0);
}

Point2D::operator Vector2D() const
{	
	return Vector2D(x, y);
}

double& Point2D::operator [] (unsigned int i)
{
	assert(i<2);
	double* p=reinterpret_cast<double*>(this);
	return p[i];
}

double Point2D::operator [] (unsigned int i) const
{
	assert(i<2);
	const double* p=reinterpret_cast<const double*>(this);
	return p[i];
}

double Point2D::Distance (const Point2D& p) const        
{	
	return(sqrt((x-p.x)*(x-p.x) + (y-p.y)*(y-p.y) )); 
}

//Normal functions.
//点加向量
Point2D operator + (const Point2D& pt, const Vector2D& v)
{
	return Point2D (pt.x + v.x, pt.y + v.y); 
}

//点减向量
Point2D operator - (const Point2D& pt, const Vector2D& v) 
{
	return Point2D (pt.x - v.x, pt.y - v.y); 
}

//两点相减
Vector2D operator - (const Point2D& p, const Point2D& q)
{
	return Vector2D (p.x - q.x, p.y - q.y); 
}

//两点相加

//数乘
Point2D operator * (const Point2D& pt, double num)

{ 
	return Point2D (pt.x * num, pt.y * num); 
}

Point2D operator * (double num, const Point2D& pt)
{
	return Point2D(num*pt.x, num*pt.y);
}
//数除
Point2D operator / (const Point2D& pt, double num) 
{ 
	//ASSERT (fabs (num) < TOLER);
	return Point2D (pt.x / num, pt.y / num); 
}

bool operator==	(const Point2D& u, const Point2D& v)
{
	return Length2(u - v) <= TOLER * TOLER;	//  [3/31/2007 YangLei]
}

bool operator!=	(const Point2D& u, const Point2D& v)
{
	return !(u==v);
}

//End Point2D
/*-------------------------------------------------------*/

//Point3D
/*-------------------------------------------------------*/

Point3D::Point3D (double newx, double newy, double newz)
:x(newx), y(newy), z(newz)
{
}

Point3D& Point3D::operator += (const Vector3D& v)
{
	x += v.x;	y += v.y;	z += v.z; 
	return *this; 
}

Point3D& Point3D::operator -= (const Vector3D& v)
{
	x -= v.x;	y -= v.y;	z -= v.z; 
	return *this; 
}

Point3D::operator Vector3D() const
{	
	return Vector3D(x, y, z);
}

Point3D Point3D::operator - () const
{
	return Point3D (-x, -y, -z); 
}

double& Point3D::operator [] (unsigned int i)
{
	assert(i<3);
	double* p=reinterpret_cast<double*>(this);
	return p[i];
}

double Point3D::operator [] (unsigned int i) const
{
	assert(i<3);
	const double* p=reinterpret_cast<const double*>(this);
	return p[i];
}

//Normal functions

//点加向量
Point3D operator +  (const Point3D& u, const Vector3D& v)
{
	return Vector3D (u.x + v.x, u.y + v.y, u.z + v.z); 
}
//点减向量
Point3D operator -  (const Point3D& u, const Vector3D& v)
{
	return Vector3D (u.x - v.x, u.y - v.y, u.z - v.z); 
}
//两点相减
Vector3D operator - (const Point3D& u, const Point3D& v)
{
	return Vector3D (u.x - v.x, u.y - v.y, u.z - v.z); 
}

//数乘
Point3D operator * (const Point3D& pt, double num)
{ 
	return Point3D (pt.x * num, pt.y * num, pt.z*num); 
}

Point3D operator * (double num, const Point3D& pt)
{
	return Point3D(num*pt.x, num*pt.y, num*pt.z);
}
//数除
Point3D operator / (const Point3D& pt, double num)
{ 
	//ASSERT (fabs (num) < TOLER);
	num = 1.0/num;
	return Point3D (pt.x * num, pt.y * num, pt.z*num); 
	//减少除法运算，LiuXiaoming 2007-6-12
}
//判等(不等)
bool operator == (const Point3D& u, const Point3D& v)
{
	return Length2(u - v) <= TOLER * TOLER;	//  [3/31/2007 YangLei]
}

bool operator != (const Point3D& u, const Point3D& v)
{
	return !(u==v);
}

//NOTE!
Vector3D operator - (const Vector3D& v, const Point3D& p)
{
	return Vector3D(v.x-p.x, v.y-p.y, v.z-p.z );
}

Vector2D operator - (const Vector2D& v, const Point2D& p)
{
	return Vector2D(v.x-p.x, v.y-p.y);
}


//End of Point3D
/*-------------------------------------------------------*/

//Point4D
/*-------------------------------------------------------*/
Point4D::Point4D (double x_, double y_, double z_, double w_)
:x(x_), y(y_), z(z_), w(w_)
{
}

Point4D& Point4D::operator += (const Point4D& v)
{ 
	x += v.x;	y += v.y;	z += v.z;	w += v.w; 
	return *this;
}

Point4D& Point4D::operator -= (const Point4D& v)
{ 
	x -= v.x;	y -= v.y;	z -= v.z;	w -= v.w; 
	return *this;
}

Point4D Point4D::operator - () const
{
	return Point4D (-x, -y, -z, -w);  
}

double& Point4D::operator [] (unsigned int i)
{
	assert(i<4);
	double* p=reinterpret_cast<double*>(this);
	return p[i];
}

double Point4D::operator [] (unsigned int i) const
{
	assert(i<4);
	const double* p=reinterpret_cast<const double*>(this);
	return p[i];
}

void Point4D::setValue(double x_, double y_, double z_, double w_)
{ 
	x = x_;		y = y_;		z = z_;		w = w_;
}

void Point4D::setValue(const Vector3D& v,double w_)
{
	x = v.x*w_;	  y = v.y * w_;	  z = v.z* w_;	 w = w_;
}

void Point4D::GetValueFrom3D(const Point3D& p, double w_)
{
	x = p.x * w_;  y = p.y * w_;  z = p.z* w_;  w = w_;
}

void Point4D::SetPoint3DValue(Point3D& p, double& w_) const
{
	w_ = w;
	double cw = 1.0/w;
	p.x = x * cw;
	p.y = y * cw;
	p.z = z * cw;
}

//Normal functions
//点加
Point4D  operator + (const Point4D& pt, const Vector4D& v)
{
	return Point4D (pt.x + v.x, pt.y + v.y, pt.z + v.z, pt.w + v.w ); 
}

Vector4D operator + (const Point4D& p, const Point4D& q)
{
	return Point4D (p.x + q.x, p.y + q.y, p.z + q.z, p.w+q.w)-Point4D(0.,0.,0.,0.); 
}

//点减
Point4D operator - (const Point4D& pt, const Vector4D& v)
{
	return Point4D (pt.x - v.x, pt.y - v.y, pt.z - v.z, pt.w - v.w); 
}
//两点相减
Vector4D operator - (const Point4D& p, const Point4D& q)
{
	return Vector4D (p.x - q.x, p.y - q.y, p.z - q.z, p.w - q.w ); 
}
//数乘
Point4D operator * (const Point4D& pt, double num) 
{ 
	return Point4D (pt.x * num, pt.y * num, pt.z*num, pt.w*num); 
}
Point4D operator * (double num, const Point4D& pt)
{
	return Point4D(num*pt.x, num*pt.y, num*pt.z, num*pt.w);
}
//数除
Point4D operator / (const Point4D& pt, double num) 
{ 
	//ASSERT (fabs (num) < TOLER);
	num  = 1.0/num;
	//return Point4D (x / num, y / num, z/num, w/num); 
	return Point4D (pt.x * num, pt.y * num, pt.z*num, pt.w*num); 
	//减少除法运算，LiuXiaoming 2007-6-12
}

//判等(不等)
bool operator == (const Point4D& p, const Point4D& q)
{
	return Length2(p-q) <= TOLER * TOLER;
}

bool operator != (const Point4D& p, const Point4D& q)
{
	return !(p==q);
}

//End of Point4D
/*-------------------------------------------------------*/

//Vector4D
/*-------------------------------------------------------*/

//构造函数
Vector4D::Vector4D (double x_, double y_, double z_, double w_)
:x(x_), y(y_), z(z_), w(w_)
{
}

Vector4D::Vector4D (const Vector3D& v, double w_)
:x(v.x), y(v.y), z(v.z), w(w_)
{
}

//Assignment operators.
Vector4D& Vector4D::operator += (const Vector4D& v)
{
	x += v.x;   y += v.y;   z += v.z;   w+=v.w;
	return *this;
}

Vector4D& Vector4D::operator -= (const Vector4D& v)
{
	x -= v.x;   y -= v.y;   z -= v.z;   w -= v.w; 
	return *this;
}

Vector4D& Vector4D::operator *= (double num)
{
	x *= num;   y *= num;   z *= num;   w *= num;
	return *this;
}

Vector4D& Vector4D::operator /= (double num)
{ 
	//ASSERT (fabs (num) < TOLER);
	num = 1.0/num;
	//x /= num; y /= num; z /= num; w /= num;
	x *= num;   y *= num;   z *= num;   w *= num; 
	//减少除法，LiuXiaoming 2007-6-12
	return *this;
}

Vector4D Vector4D::operator - () const
{
	return Vector4D (-x, -y, -z, -w); 
}

Vector4D::operator Point4D () const
{
	return Point4D (x, y, z, w);
}

void Vector4D::setValue(const Vector3D& v, double w_)
{
	x = v.x*w_; y = v.y*w_; z = v.z*w_; w = w_;
}
void Vector4D::setValue(double x_,double y_,double z_, double w_)
{
	x = x_; y = y_; z = z_; w = w_;
}

// 取长度
double Vector4D::Length ()  const                             
{ 
	return sqrt (x*x + y*y + z*z + w*w); 
}

void Vector4D::SetUnit ()
{
	double a = Length (); // * (1 + 1e-15);
	if (a > TOLER) (*this) /= a;
}

//取元素
double& Vector4D::operator [] (unsigned int i)
{
	assert(i<4);
	double *p=reinterpret_cast<double*>(this);
	return p[i];
}

double  Vector4D::operator [] (unsigned int i) const
{
	assert(i<4);
	const double *p=reinterpret_cast<const double*>(this);
	return p[i];
}


//矢量加
Vector4D operator + (const Vector4D& u, const Vector4D& v)
{ 
	return Vector4D (u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w); 
}

//矢量减
Vector4D operator - (const Vector4D& u, const Vector4D& v) 
{ 
	return Vector4D (u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w); 
}
Vector4D operator - (const Vector4D& u, const Point4D& v) 
{ 
	return Vector4D (u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w); 
}
//矢量乘(点积)
double  operator * (const Vector4D& u, const Vector4D& v) 
{ 
	return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w; 
}

//数乘
Vector4D operator * (const Vector4D& u, double num) 
{ 
	return Vector4D (u.x * num, u.y * num, u.z * num , u.w * num); 
}
//数除
Vector4D operator / (const Vector4D& u, double num) 
{ 
	//ASSERT (fabs (num) < TOLER);
	num = 1.0/num;
	return Vector4D (u.x * num, u.y * num, u.z * num , u.w * num);
}

//End of Vector4D
/*-------------------------------------------------------*/



inline double Distance (const Point2D& p1, const Point2D& p2)
{
	return Length (p1 - p2);
}

inline double Distance (const Point3D& p1, const Point3D& p2)
{
	return Length (p1 - p2);
}

inline double Distance2 (const Point2D& p1, const Point2D& p2)
{
	return Length2 (p1 - p2);
}

inline double Distance2 (const Point3D& p1, const Point3D& p2)
{
	return Length2 (p1 - p2);
}


inline Point2D MidPoint (const Point2D& p1, const Point2D& p2)
{
	return Point2D ((p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5);
}

inline Point3D MidPoint (const Point3D& p1, const Point3D& p2)
{
	return Point3D ((p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5, (p1.z + p2.z) * 0.5);
}


inline double Min(double x, double y)
{
	return ( x < y ? x:y );
}

inline double Max(double x, double y)
{
	return ( x > y ? x:y );
}

inline int Min(int x, int y)
{
	return ( x < y ? x:y );
}

inline int Max(int x, int y)
{
	return ( x > y ? x:y );
}


inline Point2D Min (const Point2D& p1, const Point2D& p2)
{
	return Point2D (Min (p1.x, p2.x), Min (p1.y, p2.y));
}

inline Point3D Min (const Point3D& p1, const Point3D& p2)
{
	return Point3D (Min (p1.x, p2.x), Min (p1.y, p2.y), Min (p1.z, p2.z));
}

inline Point2D Max (const Point2D& p1, const Point2D& p2)
{
	return Point2D (Max (p1.x, p2.x), Max (p1.y, p2.y));
}

inline Point3D Max (const Point3D& p1, const Point3D& p2)
{
	return Point3D (Max (p1.x, p2.x), Max (p1.y, p2.y), Max (p1.z, p2.z));
}

inline double Length (const Vector2D& v)
{
	return sqrt (v * v);
}

inline double Length (const Vector3D& v)
{
	return sqrt (v * v);
}

//  [3/31/2007 YangLei]
inline double Length2 (const Vector2D& v)
{
	return (v * v);
}

inline double Length2 (const Vector3D& v)
{
	return (v * v);
}

inline bool SamePoint (const Point2D& p1, const Point2D& p2, double toler)
{
//	return Distance (p1, p2) < toler;//  [3/31/2007 YangLei]
	return Length2 (p1 - p2) < toler * toler;
}

inline bool SamePoint (const Point3D& p1, const Point3D& p2, double toler)
{
//	return Distance (p1, p2) < toler;
	return Length2 (p1 - p2) < toler * toler;
}

inline bool SameVector (const Vector2D& v1, const Vector2D& v2, double toler)
{
//	return Length (v1 - v2) <= toler;
	return Length2 (v1 - v2) <= toler * toler;
}
inline bool SameVector (const Vector3D& v1, const Vector3D& v2, double toler)
{
//	return Length (v1 - v2) <= toler;
	return Length2 (v1 - v2) <= toler * toler;
}

inline void SwapPoint (Point2D& p1, Point2D& p2)
{
	Point2D p0 = p1; p1 = p2; p2 = p0;
}

inline void SwapPoint (Point3D& p1, Point3D& p2)
{
	Point3D p0 = p1; p1 = p2; p2 = p0;
}

inline void SwapVector (Vector2D& p1, Vector2D& p2)
{
	Vector2D p0 = p1; p1 = p2; p2 = p0;
}

inline void SwapVector (Vector3D& p1, Vector3D& p2)
{
	Vector3D p0 = p1; p1 = p2; p2 = p0;
}



//---------------------------------------------------------
// I/O functions
//---------------------------------------------------------
/*
#ifndef NO_MFC_SUPPORT

inline CArchive& operator << (CArchive& ar, const Point2D& aPoint)
{
	ar << aPoint.x << aPoint.y;
	return ar;
}

inline CArchive& operator >> (CArchive& ar, Point2D& aPoint)
{
	ar >> aPoint.x >> aPoint.y;
	return ar;
}

inline CArchive& operator << (CArchive& ar, const Point3D& aPoint)
{
	ar << aPoint.x << aPoint.y << aPoint.z;
	return ar;
}

inline CArchive& operator >> (CArchive& ar, Point3D& aPoint)
{
	ar >> aPoint.x >> aPoint.y >> aPoint.z;
	return ar;
}

inline CArchive& operator << (CArchive& ar, const Vector2D& aVector)
{
	ar << aVector.x << aVector.y;
	return ar;
}

inline CArchive& operator >> (CArchive& ar, Vector2D& aVector)
{
	ar >> aVector.x >> aVector.y;
	return ar;
}

inline CArchive& operator << (CArchive& ar, const Vector3D& aVector)
{
	ar << aVector.x << aVector.y << aVector.z;
	return ar;
}

inline CArchive& operator >> (CArchive& ar, Vector3D& aVector)
{
	ar >> aVector.x >> aVector.y >> aVector.z;
	return ar;
}

#endif
*/
inline std::ostream& operator<< (std::ostream& ar, const Point2D& aPoint)
{
	ar << aPoint.x << aPoint.y;
	return ar;
}

inline std::istream& operator >> (std::istream& ar,  Point2D& aPoint)
{
	ar >> aPoint.x >> aPoint.y;
	return ar;
}

inline std::ostream& operator << (std::ostream& ar, const Point3D& aPoint)
{
	ar << aPoint.x << aPoint.y << aPoint.z;
	return ar;
}

inline std::istream& operator >> (std::istream& ar, Point3D& aPoint)
{
	ar >> aPoint.x >> aPoint.y >> aPoint.z;
	return ar;
}

inline std::ostream& operator << (std::ostream& ar, const Vector2D& aVector)
{
	ar << aVector.x << aVector.y;
	return ar;
}

inline std::istream& operator >> (std::istream& ar, Vector2D& aVector)
{
	ar >> aVector.x >> aVector.y;
	return ar;
}

inline std::ostream& operator << (std::ostream& ar, const Vector3D& aVector)
{
	ar << aVector.x << aVector.y << aVector.z;
	return ar;
}

inline std::istream& operator >> (std::istream& ar, Vector3D& aVector)
{
	ar >> aVector.x >> aVector.y >> aVector.z;
	return ar;
}

//--------------------------------------------------------------------------------
inline double Distance (const Point4D& p1, const Point4D& p2)
{
	return sqrt(	(p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) +
					(p1.z-p2.z)*(p1.z-p2.z) + (p1.w-p2.w)*(p1.w-p2.w)
				);
}

inline double Distance2 (const Point4D& p1, const Point4D& p2)
{
	return ( (p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) +
			 (p1.z-p2.z)*(p1.z-p2.z) + (p1.w-p2.w)*(p1.w-p2.w) 
		   );
}

inline double Length (const Vector4D& v)
{
	return sqrt (v * v);
}

inline double Length2 (const Vector4D& v)
{
	return (v * v);
}

inline bool SamePoint (const Point4D& p1, const Point4D& p2, double toler)
{
	return Length2(p1 - p2) < toler * toler;	//  [3/31/2007 YangLei]
}
/*-------------------------------------------------------*/

} // namespace icesop
