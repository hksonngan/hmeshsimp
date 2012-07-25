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


#ifndef _MATHLIB_H
#define _MATHLIB_H

#include "../datastructures/point.h"

namespace icesop {

#define MATH_DECLSPECIFIER

// 数学库的头文件

// **************************************** // 
//             Hu Shi-Min 
// **************************************** //

//1.返回二次方程的实根
MATH_DECLSPECIFIER int Equat2(double ta,double tb,double tc,double &s1, double &s2, double toler = TOLER);
//2. 返回三次方程的实根
MATH_DECLSPECIFIER int Equat3(double ta, double tb, double tc, double td, double &h1, double &h2, double &h3, double toler = TOLER);
//3. 返回四次方程的实根
MATH_DECLSPECIFIER int Equat4(double ha,double hb,double hc,double hd,double he,double &h1,double &h2,double &h3,double &h4, double toler = TOLER);
//4. 根排序和剔除重根 , 用于Equat4();
MATH_DECLSPECIFIER int RootSort(int r1, int r2, double &rs1, double &rs2, double &rs3, double &rs4, double toler = TOLER); 
//5. 返回高次方程的实根   coef[0]*x^n + coef[1]*x^{n-1} + ... + coef[0] = 0
MATH_DECLSPECIFIER int Equatn(int degree, double coef[], double root[], double toler = TOLER);
//利用求导的思想解高次一元方程,解4次以上(不含4次)的方程:added by lxz:2003-11-17
MATH_DECLSPECIFIER int Equatn(int degree, double coef[], double root[],double low,double upper,double toler = TOLER,double coefTol = 1e-60);
//利用求导的思想解高次一元方程,解4次以上(不含4次)的方程，不需要区间；added by lxz:2003-11-17
MATH_DECLSPECIFIER int EquatnNew(int degree, double coef[], double root[],double toler = TOLER,double coefTol = 1e-60);
//二分法求解方程:added by lxz:2003-11-17
MATH_DECLSPECIFIER bool  binarySolution(int degree,double coef[],double left,double right,double &val,double toler = TOLER,double coefTol = 1e-60);
//以下三个方程可以返回二分法的迭代步数:added by lxz:2003-11-17
MATH_DECLSPECIFIER int EquatnNew(int degree, double coef[], double root[],int& step,double toler = TOLER);
MATH_DECLSPECIFIER int Equatn(int degree, double coef[], double root[],double low,double upper,int &step,double toler = TOLER);
MATH_DECLSPECIFIER bool  binarySolution(int degree,double coef[],double left,double right,double &val,int &step,double toler = TOLER);
//
//6. 多项式消去因子（x-fa), degree 为多项式的次数，coef为系数
MATH_DECLSPECIFIER void reduce_single(int degree, double coef[], double fa);
//7. 多项式消去因子（x^2+fa*x+fb), degree 为多项式的次数，coef为系数
MATH_DECLSPECIFIER void reduce_twice(int degree, double coef[], double fa, double fb);
//8. 剔除重复者
MATH_DECLSPECIFIER void Sort_reduce(int &rn, double rt[]);
//9. Newton法求代数方程的根：次数 degree, 系数：coef, 初值：start_value, 传引用返回根 root
MATH_DECLSPECIFIER int Newton(int degree,double coef[],double start_value,double &root);
// 10. 用于Newton(): 计算f(x_k) and f'(x_k) 
MATH_DECLSPECIFIER void NewtonAided(int num, double a[], double x, double &fx, double &dfx);
// 11. Bernouli法求代数方程的根： 次数 degree, 系数：coef, 传引用返回根 root
MATH_DECLSPECIFIER int Bernouli(int degree, double coefficent[], double &root, double &p, double &q);

//12 .求ef的三次方根
MATH_DECLSPECIFIER double Evolut3(double ef);
// 13. 求exponent个cardinal_numb相乘:  cardinal^exponent       
MATH_DECLSPECIFIER double Power(double cardinal_numb,int exponent);
// 14. 多项式求值
MATH_DECLSPECIFIER double PolynomialValue(int n, double a[], double x);      
// 15. 求反余弦函数
MATH_DECLSPECIFIER double getacos(double value);
// 16. 求正余弦函数
MATH_DECLSPECIFIER double getasin(double value);
       


// 101 求解线性方程组 (右端为数) 
// a为系数矩阵，degree 为矩阵的阶数
MATH_DECLSPECIFIER int EquatSystemNumber(int degree, double **a, double *y, double *x);
// 102 求解线性方程组 (右端为三维点) 
// a为系数矩阵，degree 为矩阵的阶数
MATH_DECLSPECIFIER int EquatSystemPoint(int degree, double **a, Point3D *y, Point3D *x);

// Zuo Zheng
// 200. 求一次三角函数Y=Acosx+Bsinx(a<=x<=b)的最小最大值
MATH_DECLSPECIFIER void CalTriangleMinMax(double A, double B, double a, double b, double &Ymin, double &Ymax);

/* ****************************************************************
	求解带状线性方程组的函数
	ripped from the Numerical Recipes
	详细原理参见NR一书
	Yang Lei 06-01-09
*/

// 带状线性方程组的高斯消元法
MATH_DECLSPECIFIER void bandec(double **a, int n, int m1, int m2, double **al,
							   int indx[], double *d);

// 高斯消元后的回填（back substitution）
MATH_DECLSPECIFIER void banbks(double **a, int n, int m1, int m2, double **al,
							   int indx[], double b[]);

// 带状矩阵向量相乘
MATH_DECLSPECIFIER void banmul(double **a, int n, int m1, int m2, double x[], double b[]);

} // namespace icesop

#endif     
