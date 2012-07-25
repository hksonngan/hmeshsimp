/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** �����˶�ά����ά����άʸ���͵�:
** ȡ��Gems7�ں����ݽṹ����
** �ɶ�ʸ���͵���и�ֵ���Ӽ�����˳�����ˡ��Ȳ���
** ����, ���ɶ�ʸ�����е�λ����ȡ���ȵȲ���
**
** Author : shan @2011
**
****************************************************************************/

#ifndef ICESOP_POINT_H
#define ICESOP_POINT_H

#include <cmath>

namespace icesop {

#define PI2			6.28318530717958647692
#define PI			3.14159265358979323846
#define PI_2        1.57079632679489661923
#define PI_4        0.785398163397448309616
#define PI_8        0.392699081698724154808
#define PI_16       0.196349540849362077404

//NOTICE: 
/**
__declspec specifier has been removed from this file since it's fully inlined.

**/

#define M_E         2.71828182845904523536
#define M_LOG2E     1.44269504088896340736
#define M_LOG10E    0.434294481903251827651
#define M_LN2       0.693147180559945309417
#define M_LN10      2.30258509299404568402
#define M_SQRT2     1.41421356237309504880
#define M_SQRT2_2   0.707106781186547524401

#define TOLER		1e-10

class Vector2D;
class Vector3D;
class Vector4D;
class Point2D;
class Point3D;
class Point4D;

/*-------------------------------------------------------*/

class Vector2D
{
public:
	double	x, y;

public:
	//constructors
	explicit inline Vector2D (double newx=0.0, double newy=0.0);

	//assignment operators
	inline Vector2D& operator += (const Vector2D& v);
	inline Vector2D& operator -= (const Vector2D& v);
	inline Vector2D& operator *= (double num);
	inline Vector2D& operator /= (double num);
	
	//��Ŀ��
	inline Vector2D operator - () const;
	
	//����ת��
	inline operator Point2D () const;
	inline Vector3D ConvertTo3D() const;
	inline Vector2D perpVector() const;

	//ȡԪ��
	inline double& operator [] (unsigned int i);
	inline double operator [] (unsigned int i) const;

	// ȡ����
    inline double Length ()  const;

	//��Ϊ��λʸ��
	inline void SetUnit();

};


inline bool operator == (const Vector2D& u, const Vector2D& v);	// ���ȽϷ��򣬺��Գ���
inline bool operator != (const Vector2D& u, const Vector2D& v);	// ���ȽϷ��򣬺��Գ���


//bi-ops
inline double operator ^ (const Vector2D& u, const Vector2D& v);
//ʸ����
inline Vector2D operator + (const Vector2D& u, const Vector2D& v); 
//ʸ����
inline Vector2D operator - (const Vector2D& u, const Vector2D& v); 
//ʸ����(���)
inline double  operator * (const Vector2D& u, const Vector2D& v); 
//����
inline Vector2D operator * (const Vector2D& v, double num);
inline Vector2D operator * (double num, const Vector2D& v);
//����
inline Vector2D operator / (const Vector2D& v, double num); 


/*-------------------------------------------------------*/

class	Vector3D
{
public:
	double	x, y, z;

public:
//constructors
	explicit inline Vector3D (double newx=0.0, double newy=0.0, double newz=0.0);

	//��ֵ������
	inline Vector3D& operator += (const Vector3D& v);
	inline Vector3D& operator -= (const Vector3D& v);
	inline Vector3D& operator *= (double num);
	inline Vector3D& operator /= (double num);
	inline Vector3D& operator ^= (const Vector3D& v);

	//��Ŀ��
	inline Vector3D operator - () const;

	//����ת��
	inline operator Point3D () const;
	inline Vector3D perpVector() const;

	//ȡԪ��
	inline double& operator [] (unsigned int i);
	inline double operator[] (unsigned int i) const;

	// ȡ����
    inline double Length ()  const;

	//��Ϊ��λʸ��
	inline void SetUnit ();
	inline void setValue(double newx=0.0, double newy=0.0,double newz=0.0);
};

//ʸ����
inline Vector3D operator + (const Vector3D& u, const Vector3D& v) ;

//ʸ����
inline Vector3D operator - (const Vector3D& u, const Vector3D& v) ;

//ʸ����(���)
inline double operator * (const Vector3D& u, const Vector3D& v) ;
//ʸ����(���)
inline Vector3D operator ^ (const Vector3D& u, const Vector3D& v); 

//����
inline Vector3D operator * (const Vector3D& v, double num);
inline Vector3D operator * (double num, const Vector3D& v); 
//����
inline Vector3D operator / (const Vector3D& v, double num); 

inline bool operator == (const Vector3D& u, const Vector3D& v);	
// ���ȽϷ��򣬺��Գ���
inline bool operator != (const Vector3D& u, const Vector3D& v);	
// ���ȽϷ��򣬺��Գ���
/*-------------------------------------------------------*/

class Point2D
{
public:
	double	x, y;
public:
	//���캯��
	explicit inline Point2D (double newx=0.0, double newy=0.0);
	
	inline Point2D& operator += (const Vector2D& v);
	inline Point2D& operator -= (const Vector2D& v);

	//��Ŀ��
	inline Point2D operator - () const;
	
	//����ת��
    inline Point3D ConvertTo3D() const;
	inline operator Vector2D () const;
 
    //ȡԪ��
	inline double& operator [] (unsigned int i);
	inline double operator[] (unsigned int i) const;

	//--------------------------------------------------------------------------
	//Add by Peng Yu 2003-11-22
	inline double Distance (const Point2D& p) const;
	//--------------------------------------------------------------------------
};

//�������
inline Point2D operator + (const Point2D& pt, const Vector2D& v);
//�������
inline Point2D operator - (const Point2D& pt, const Vector2D& v);
//�������
inline Vector2D operator - (const Point2D& p, const Point2D& q);
//�������

//����
inline Point2D operator * (const Point2D& pt, double num);
inline Point2D operator * (double num, const Point2D& pt);
//����
inline Point2D operator / (const Point2D& pt, double num);

//�е�(����)
inline	bool operator == (const Point2D& u, const Point2D& v) ;
inline	bool operator != (const Point2D& u, const Point2D& v) ;

/*-------------------------------------------------------*/


class Point3D
{
public:
	double	x, y, z;
public:
	//���캯��
	explicit inline Point3D (double newx=0.0, double newy=0.0, double newz=0.0);

	//��ֵ������
	inline Point3D& operator += (const Vector3D& v);
	inline Point3D& operator -= (const Vector3D& v);

	//��Ŀ��
	inline Point3D operator - () const;
 	//����ת��
	inline operator Vector3D () const;

	//ȡԪ��
	inline double& operator [] (unsigned int i);
	inline double  operator [] (unsigned int i) const;
};

//�������
inline Point3D operator +  (const Point3D& u, const Vector3D& v);
//�������
inline Point3D operator -  (const Point3D& u, const Vector3D& v);
//�������
inline Vector3D operator - (const Point3D& u, const Point3D& v);
//����
inline Point3D operator * (const Point3D& pt, double num);
inline Point3D operator * (double num, const Point3D& pt);
//����
inline Point3D operator / (const Point3D& pt, double num);
//�е�(����)
inline bool operator == (const Point3D& u, const Point3D& v);
inline bool operator != (const Point3D& u, const Point3D& v);


//NOTE!
inline Vector3D operator - (const Vector3D& v, const Point3D& p);
inline Vector2D operator - (const Vector2D& v, const Point2D& p);


class Point4D
{
public:
	double	x, y, z, w;
public:
	//���캯��
	explicit inline Point4D (double x_=0., double y_=0., double z_=0., double w_=0.);
	
	inline Point4D& operator += (const Point4D& v);
	inline Point4D& operator -= (const Point4D& v);

	//��Ŀ��
	inline Point4D operator - () const;

	//ȡԪ��
	inline double& operator [] (unsigned int i);
	inline double  operator [] (unsigned int i) const;

	//��ֵ
	inline void setValue(double x_=0., double y_=0., double z_=0., double w_=0.);
	inline void setValue(const Vector3D& v,double w_);

	//added by LiuXiaoming 2007-6-7
	//��ά�ռ�����ά�����������໥ת��
	inline void GetValueFrom3D(const Point3D& p, double w_);
	inline void SetPoint3DValue(Point3D& p, double& weight) const;
};

//���
inline Point4D  operator + (const Point4D& pt, const Vector4D& v);
inline Vector4D operator + (const Point4D& p, const Point4D& q);
//���
inline Point4D  operator - (const Point4D& pt, const Vector4D& v);
//�������
inline Vector4D operator - (const Point4D& p, const Point4D& q);
//����
inline Point4D operator * (const Point4D& pt, double num); 
inline Point4D operator * (double num, const Point4D& pt);
//����
inline Point4D operator / (const Point4D& pt, double num);
//�е�(����)
inline bool operator == (const Point4D& p, const Point4D& q);
inline bool operator != (const Point4D& p, const Point4D& q);



class	Vector4D
{
public:
	double	x, y, z, w;
public:
	//���캯��
	explicit inline Vector4D (double x_=0., double y_=0., double z_=0., double w_=0.);
	inline Vector4D (const Vector3D& v, double w_);

	//Assignment operators.
	inline Vector4D& operator += (const Vector4D& v);
	inline Vector4D& operator -= (const Vector4D& v);
	inline Vector4D& operator *= (double num);
	inline Vector4D& operator /= (double num);
		
	//��Ŀ��
	inline Vector4D operator - () const;
	//����ת��
	inline operator Point4D () const;

	inline void setValue(const Vector3D& v, double w_);
	inline void setValue(double x_,double y_,double z_, double w_);
	inline void SetUnit();

	// ȡ����
    inline double Length ()  const;

	//ȡԪ��
	inline double& operator [] (unsigned int i);
	inline double  operator [] (unsigned int i) const;
};

//ʸ����
inline Vector4D operator + (const Vector4D& u, const Vector4D& v);
//ʸ����
inline Vector4D operator - (const Vector4D& u, const Vector4D& v); 
inline Vector4D operator - (const Vector4D& u, const Point4D& v); 
//ʸ����(���)
inline double  operator * (const Vector4D& u, const Vector4D& v); 
//����
inline Vector4D operator * (const Vector4D& u, double num); 
//����
inline Vector4D operator / (const Vector4D& u, double num); 


//Normal functions

inline double Distance (const Point2D& p1, const Point2D& p2);
inline double Distance (const Point3D& p1, const Point3D& p2);
inline double Distance (const Point4D& p1, const Point4D& p2);
//  [3/31/2007 YangLei] ƽ������
inline double Distance2 (const Point2D& p1, const Point2D& p2);
inline double Distance2 (const Point3D& p1, const Point3D& p2);
inline double Distance2 (const Point4D& p1, const Point4D& p2);

inline Point2D MidPoint (const Point2D& p1, const Point2D& p2);
inline Point3D MidPoint (const Point3D& p1, const Point3D& p2);

inline Point2D Min (const Point2D& p1, const Point2D& p2);
inline Point3D Min (const Point3D& p1, const Point3D& p2);
inline Point2D Max (const Point2D& p1, const Point2D& p2);
inline Point3D Max (const Point3D& p1, const Point3D& p2);

inline double Length (const Vector2D& v);
inline double Length (const Vector3D& v);
inline double Length (const Vector4D& v);
// [3/31/2007 YangLei] ƽ������
inline double Length2 (const Vector2D& v);
inline double Length2 (const Vector3D& v);
inline double Length2 (const Vector4D& v);

inline bool SamePoint (const Point2D& p1, const Point2D& p2, double toler = TOLER);
inline bool SamePoint (const Point3D& p1, const Point3D& p2, double toler = TOLER);
inline bool SamePoint(const Point4D& p1, const Point4D& p2, double toler = TOLER);	//  [3/31/2007 YangLei]

// �����뷽������򷵻�true
inline bool SameVector (const Vector2D& v1, const Vector2D& v2, double toler = TOLER);
inline bool SameVector (const Vector3D& v1, const Vector3D& v2, double toler = TOLER);

inline void SwapPoint (Point2D& p1, Point2D& p2);
inline void SwapPoint (Point3D& p1, Point3D& p2);
inline void SwapVector (Vector2D& p1, Vector2D& p2);
inline void SwapVector (Vector3D& p1, Vector3D& p2);

inline double Min(double x, double y);

inline double Max(double x, double y);

inline int Min(int x, int y);

inline int Max(int x, int y);

#define ZeroP2D Point2D(0.0, 0.0)
#define ZeroP3D Point3D(0.0, 0.0, 0.0)
#define ZeroP4D Point4D(0.0, 0.0, 0.0, 0.0)
       
#define BaseXV2D Vector2D(1.0, 0.0)
#define BaseYV2D Vector2D(0.0, 1.0)
        
#define BaseXV3D Vector3D(1.0, 0.0, 0.0)
#define BaseYV3D Vector3D(0.0, 1.0, 0.0)
#define BaseZV3D Vector3D(0.0, 0.0, 1.0)


} // namespace icesop

#include "Point.inl"

#endif // ICESOP_POINT_H
