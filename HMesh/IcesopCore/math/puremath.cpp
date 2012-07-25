/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ������ѧ������
** ȡ��Gems7�ں�MathDll����
**
** Author : shan @2011
**
****************************************************************************/

#include "mathlib.h"
#include <cmath>
#include <algorithm>

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

namespace icesop {

// **************************************** // 
//             Hu Shi-Min 
// **************************************** //

            


//===============================================================================
//1 .��ef�����η���
double Evolut3(double ef)
{  
   double vev;
   if( fabs(ef) < TOLER )
       vev = 0.0; 
   else if( ef > 0 )     
       vev = exp(log(ef)/3.0);
   else 
       vev = - exp(log(0.0 - ef)/3.0);
   return(vev);
}


    
//===============================================================================
// 2. ��exponent��cardinal_numb���       
double Power(double cardinal_numb,int exponent)
// Computing  cardinal^exponent                                            
{
    double result = 1;
    for( int i = 1; i <= exponent; i++ )
        result = result*cardinal_numb;
    return result;
} 

                                   
    
//===============================================================================
// 3. ����ʽ��ֵ
double PolynomialValue(int n, double a[], double x)
{  
    double f = a[0];
    for( int i = 1; i <= n; i++)
        f = a[i] + f*x;            
    return(f);
}

    
//===============================================================================
// 4. �����Һ���
double getacos(double value)
{
	if(value - TOLER <= -1.0)
		return PI;
	if(value + TOLER >= 1.0)
		return 0.0;
	if(fabs(value) < TOLER)
		return PI_2;
	return acos(value);
}



//===============================================================================
// 5. �����Һ���
double getasin(double value)
{
	if(value - TOLER <= -1.0)
		return PI*1.5;
	if(value + TOLER >= 1.0)
		return PI_2;
	if(fabs(value) < TOLER)
		return 0;
	return asin(value);
}


//===============================================================================
// 200. ��һ�����Ǻ���Y=Acosx+Bsinx(a<=x<=b)����С���ֵ
void CalTriangleMinMax(double A, double B, double a, double b, double &Ymin, double &Ymax)
{
    if (fabs(A) < TOLER && fabs(B) < TOLER)
        Ymin = Ymax = 0.0;
    else
    {
        double y1 = A*cos(a) + B*sin(a);
        double y2 = A*cos(b) + B*sin(b);
        double C = sqrt(A*A + B*B);
        double theta = atan2(A, B);
        if (a+theta < -PI_2 && -PI_2 < b+theta || a+theta < PI*1.5 && PI*1.5 < b+theta)
            Ymin = -C;
        else
            Ymin = std::min(y1, y2);
        if (a+theta < PI_2 && PI_2 < b+theta || a+theta < PI*2.5 && PI*2.5 < b+theta)
            Ymax = C;
        else
            Ymax = std::max(y1, y2);
    }
}

} // namespace icesop

