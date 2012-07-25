/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 用于数学函数库
** 取自Gems7内核MathDll代码
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
//1 .求ef的三次方根
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
// 2. 求exponent个cardinal_numb相乘       
double Power(double cardinal_numb,int exponent)
// Computing  cardinal^exponent                                            
{
    double result = 1;
    for( int i = 1; i <= exponent; i++ )
        result = result*cardinal_numb;
    return result;
} 

                                   
    
//===============================================================================
// 3. 多项式求值
double PolynomialValue(int n, double a[], double x)
{  
    double f = a[0];
    for( int i = 1; i <= n; i++)
        f = a[i] + f*x;            
    return(f);
}

    
//===============================================================================
// 4. 求反余弦函数
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
// 5. 求反正弦函数
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
// 200. 求一次三角函数Y=Acosx+Bsinx(a<=x<=b)的最小最大值
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

