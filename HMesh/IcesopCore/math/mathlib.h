/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ����3*3��4*4��double���ͷ���:
** ȡ��Gems7�ں����ݽṹ����
** �ɶԾ�����и�ֵ���Ӽ������˳�������˳�������Ȳ���
** ����, �������˶�ά������3*3������ά������4*4�������˲���
**
** Author : shan @2011
**
****************************************************************************/


#ifndef _MATHLIB_H
#define _MATHLIB_H

#include "../datastructures/point.h"

namespace icesop {

#define MATH_DECLSPECIFIER

// ��ѧ���ͷ�ļ�

// **************************************** // 
//             Hu Shi-Min 
// **************************************** //

//1.���ض��η��̵�ʵ��
MATH_DECLSPECIFIER int Equat2(double ta,double tb,double tc,double &s1, double &s2, double toler = TOLER);
//2. �������η��̵�ʵ��
MATH_DECLSPECIFIER int Equat3(double ta, double tb, double tc, double td, double &h1, double &h2, double &h3, double toler = TOLER);
//3. �����Ĵη��̵�ʵ��
MATH_DECLSPECIFIER int Equat4(double ha,double hb,double hc,double hd,double he,double &h1,double &h2,double &h3,double &h4, double toler = TOLER);
//4. ��������޳��ظ� , ����Equat4();
MATH_DECLSPECIFIER int RootSort(int r1, int r2, double &rs1, double &rs2, double &rs3, double &rs4, double toler = TOLER); 
//5. ���ظߴη��̵�ʵ��   coef[0]*x^n + coef[1]*x^{n-1} + ... + coef[0] = 0
MATH_DECLSPECIFIER int Equatn(int degree, double coef[], double root[], double toler = TOLER);
//�����󵼵�˼���ߴ�һԪ����,��4������(����4��)�ķ���:added by lxz:2003-11-17
MATH_DECLSPECIFIER int Equatn(int degree, double coef[], double root[],double low,double upper,double toler = TOLER,double coefTol = 1e-60);
//�����󵼵�˼���ߴ�һԪ����,��4������(����4��)�ķ��̣�����Ҫ���䣻added by lxz:2003-11-17
MATH_DECLSPECIFIER int EquatnNew(int degree, double coef[], double root[],double toler = TOLER,double coefTol = 1e-60);
//���ַ���ⷽ��:added by lxz:2003-11-17
MATH_DECLSPECIFIER bool  binarySolution(int degree,double coef[],double left,double right,double &val,double toler = TOLER,double coefTol = 1e-60);
//�����������̿��Է��ض��ַ��ĵ�������:added by lxz:2003-11-17
MATH_DECLSPECIFIER int EquatnNew(int degree, double coef[], double root[],int& step,double toler = TOLER);
MATH_DECLSPECIFIER int Equatn(int degree, double coef[], double root[],double low,double upper,int &step,double toler = TOLER);
MATH_DECLSPECIFIER bool  binarySolution(int degree,double coef[],double left,double right,double &val,int &step,double toler = TOLER);
//
//6. ����ʽ��ȥ���ӣ�x-fa), degree Ϊ����ʽ�Ĵ�����coefΪϵ��
MATH_DECLSPECIFIER void reduce_single(int degree, double coef[], double fa);
//7. ����ʽ��ȥ���ӣ�x^2+fa*x+fb), degree Ϊ����ʽ�Ĵ�����coefΪϵ��
MATH_DECLSPECIFIER void reduce_twice(int degree, double coef[], double fa, double fb);
//8. �޳��ظ���
MATH_DECLSPECIFIER void Sort_reduce(int &rn, double rt[]);
//9. Newton����������̵ĸ������� degree, ϵ����coef, ��ֵ��start_value, �����÷��ظ� root
MATH_DECLSPECIFIER int Newton(int degree,double coef[],double start_value,double &root);
// 10. ����Newton(): ����f(x_k) and f'(x_k) 
MATH_DECLSPECIFIER void NewtonAided(int num, double a[], double x, double &fx, double &dfx);
// 11. Bernouli����������̵ĸ��� ���� degree, ϵ����coef, �����÷��ظ� root
MATH_DECLSPECIFIER int Bernouli(int degree, double coefficent[], double &root, double &p, double &q);

//12 .��ef�����η���
MATH_DECLSPECIFIER double Evolut3(double ef);
// 13. ��exponent��cardinal_numb���:  cardinal^exponent       
MATH_DECLSPECIFIER double Power(double cardinal_numb,int exponent);
// 14. ����ʽ��ֵ
MATH_DECLSPECIFIER double PolynomialValue(int n, double a[], double x);      
// 15. �����Һ���
MATH_DECLSPECIFIER double getacos(double value);
// 16. �������Һ���
MATH_DECLSPECIFIER double getasin(double value);
       


// 101 ������Է����� (�Ҷ�Ϊ��) 
// aΪϵ������degree Ϊ����Ľ���
MATH_DECLSPECIFIER int EquatSystemNumber(int degree, double **a, double *y, double *x);
// 102 ������Է����� (�Ҷ�Ϊ��ά��) 
// aΪϵ������degree Ϊ����Ľ���
MATH_DECLSPECIFIER int EquatSystemPoint(int degree, double **a, Point3D *y, Point3D *x);

// Zuo Zheng
// 200. ��һ�����Ǻ���Y=Acosx+Bsinx(a<=x<=b)����С���ֵ
MATH_DECLSPECIFIER void CalTriangleMinMax(double A, double B, double a, double b, double &Ymin, double &Ymax);

/* ****************************************************************
	����״���Է�����ĺ���
	ripped from the Numerical Recipes
	��ϸԭ��μ�NRһ��
	Yang Lei 06-01-09
*/

// ��״���Է�����ĸ�˹��Ԫ��
MATH_DECLSPECIFIER void bandec(double **a, int n, int m1, int m2, double **al,
							   int indx[], double *d);

// ��˹��Ԫ��Ļ��back substitution��
MATH_DECLSPECIFIER void banbks(double **a, int n, int m1, int m2, double **al,
							   int indx[], double b[]);

// ��״�����������
MATH_DECLSPECIFIER void banmul(double **a, int n, int m1, int m2, double x[], double b[]);

} // namespace icesop

#endif     
