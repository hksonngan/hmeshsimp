/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 用于曲线曲面求交的数学函数库
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

// Used by Hu Shimin 
#define  ITERAT_NUM 124      // Newton法与Bernouli法的迭代次数
#define  DEGREE_NUM 12       // 求根的代数方程的最高次数  



// **************************************** // 
//             Hu Shi-Min 
// **************************************** //

//===============================================================================
//1.返回二次方程的实根
int Equat2(double ta,double tb,double tc,double &s1, double &s2, double toler)
{
	double tmax = std::max( fabs(ta), std::max(fabs(tb), fabs(tc)) );
    tmax = sqrt(tmax);

    if( fabs(ta)/tmax < toler )
    { 
        if( fabs(tb)/tmax > toler )
        {
            s1 = s2 = -tc/tb;
            return 1;
        }
        else
           return 0;
    }
    else
    {
        // 注意: 两根的顺序不能变
        tb /= fabs(ta);
        tc /= fabs(ta);
        ta = ta>0 ? 1.0 : -1.0;
        double td = tb*tb-4*ta*tc;
        if ( fabs(td) < toler )
        {  
            s1 = s2 = -tb/(2.0*ta);
            return 1;
        }   
        else if( td < 0)
            return 0;
        else
        {
            s1 = (-tb - sqrt(td))/(2.0*ta);
            s2 = (-tb + sqrt(td))/(2.0*ta);
            return 2;
        }
    }
}



// 2. 返回三次方程的实根
int Equat3(double ta, double tb, double tc, double td, double &h1, double &h2, double &h3, double toler)
{
    int resu;
    double p,q,delta,len,ang;

	double tmax = std::max( std::max(fabs(ta), fabs(tb)), std::max(fabs(tc), fabs(td)) );
    tmax = Evolut3(tmax);

    if( fabs(ta)/tmax < toler )    //三次项系数为零
    {
        resu = Equat2(tb,tc,td,h1,h2);
        return resu;
    }   
    else if( fabs(td)/tmax < toler )
    { 
        double h[4];   
        h[0] = 0; 
        resu = Equat2(ta,tb,tc,h[1],h[2]) + 1;  
        Sort_reduce(resu,h);
        if( resu == 1 )
        {
            h1 = h[0];
            return 1;
        }   
        else if( resu == 2 )
        {  
            h1 = h[0];   
            h2 = h[1];  
            return 2;
        }
        else 
        {  
            h1 = h[0];   
            h2 = h[1];  
            h3 = h[2];   
            return 3;
        } 
    }         
    else
    {
        tb = tb/ta;  tc = tc/ta;   td = td/ta;   ta = 1;
        p = tc - tb*tb/3.0; 
        q = 2*tb*tb*tb/27.0 + td - tb*tc/3.0;
        delta = q*q/4.0 + p*p*p/27.0; 
   
        /*if( fabs(delta) <= toler ) 
        {
            if( fabs(q) < toler )
            {  
                h1 = 0.0  - tb/3.0;   //因为有一个变换，此变换消去二次项
                return 1;
            }
            else 
            {
                h1 = 2.0*Evolut3(-q/2.0) - tb/3.0;    
                h2 = 0.0 - Evolut3(-q/2.0) - tb/3.0;   
                return 2;
            }
        }
        else if( delta > toler )  
        {
            h1 = Evolut3(-q/2.0 + sqrt(delta)) 
               + Evolut3(-q/2.0 - sqrt(delta)) - tb/3.0;
            return 1;  
        }
        else
        { 
            len = sqrt(- p*p*p/27.0);  
            ang = getacos(-q/(2*len))/3.0; 
            h1 = 2.0*Evolut3(len)*cos(ang) - tb/3.0;
            h2 = 2.0*Evolut3(len)*cos(ang + PI2/3.0) - tb/3.0;
            h3 = 2.0*Evolut3(len)*cos(ang + 4.0*PI/3.0) - tb/3.0;
            return 3;
        }*/

        double hh[4], r, hi, f0, f, fd;
        int hn, i, times;
        if( fabs(delta) <= toler ) 
        {
            if( fabs(q) < toler )
            {
                hh[0] = -tb/3.0;
                hn = 1;
            }
            else 
            {
                hh[0] = 2.0*Evolut3(-q/2.0) - tb/3.0;    
                hh[1] = -Evolut3(-q/2.0) - tb/3.0;
                hn = 2;
            }
        }
        else if( delta > toler )  
        {
            r = sqrt(delta);
            hh[0] = Evolut3(-q/2.0 + r) + Evolut3(-q/2.0 - r) - tb/3.0;
            hn = 1;  
        }
        else
        { 
            len = sqrt(- p*p*p/27.0);
            ang = getacos(-q/(2*len))/3.0;
            r = 2.0*Evolut3(len);
            hh[0] = r*cos(ang) - tb/3.0;
            hh[1] = r*cos(ang + PI2/3.0) - tb/3.0;
            hh[2] = r*cos(ang + 4.0*PI/3.0) - tb/3.0;
            hn = 3;
        }
        for (i=0; i<hn; i++)
        {
            hi = hh[i];
            times = 0;
            f0 = f = hi*hi*hi + tb*hi*hi + tc*hi + td;
            while (fabs(f) > toler && times < 20)
            {
                fd = 3*hi*hi + 2*tb*hi + tc;
				if ( fabs(fd) < TOLER )
					break;
                hi -= f / fd;
                f = hi*hi*hi + tb*hi*hi + tc*hi + td;
                times ++;
            }
            if (fabs(f) < fabs(f0))
                hh[i] = hi;
        }
        h1 = hh[0];
        if (hn > 1)
            h2 = hh[1];
        if (hn > 2)
            h3 = hh[2];
        return hn;
    }
} 
 
 

//===============================================================================
//3. 返回四次方程的实根
int Equat4(double ha,double hb,double hc,double hd,double he,double &h1,double &h2,double &h3,double &h4, double toler)
{ 
  int res,res1,res2;

  double maxh1 = std::max(fabs(hb), fabs(hc));
  double maxh2 = std::max(fabs(hd), fabs(he));
  double maxh3 = std::max( fabs(ha), std::max(maxh1, maxh2) );
  double hmax = sqrt( sqrt(maxh3) );

  if ( fabs(ha)/hmax < toler )
  {
     res = Equat3(hb,hc,hd,he,h1,h2,h3,toler);
     return res;
  }

  if( (fabs(hb)/hmax < toler) && (fabs(hd)/hmax < toler) )
  {  
     double s1,s2;
     res = Equat2(ha,hc,he,s1,s2,toler);
     if( res == 0)
        return 0;
     else if( res == 1 )
     {  
        if( s1 < - toler )
           return 0;
        else if( fabs(s1) < toler )
        {   h1 = 0;   return 1;   }
        else
        {
           h2 = sqrt(s1);    h1 = -h2;
           return 2;
        }   
     } 
     else 
     {     
        res1 = Equat2(1,0,-s1,h1,h2);
        res2 = Equat2(1,0,-s2,h3,h4);
        int resu = RootSort(res1,res2,h1,h2,h3,h4,toler);
        return resu;
     }
  }
  else if( (fabs(hb - hd)/hmax < toler) && (fabs(ha - he)/hmax < toler) )  
  {  
     double mdelta,m1,m2;
     mdelta = hb*hb - 4*ha*hc + 8*ha*ha; 
     if( mdelta < 0 ) 
        return 0; 
     else if( fabs(mdelta)/hmax < toler )
     {
        m1 = -hb/(2*ha);
        int resu = Equat2(1,-m1,1,h1,h2,toler);
        return resu;
     }    
     else
     {
        m1 = (-hb + sqrt(mdelta))/(2*ha); 
        m2 = (-hb - sqrt(mdelta))/(2*ha); 
        res1 = Equat2(1,-m1,1,h1,h2,toler);
        res2 = Equat2(1,-m2,1,h3,h4,toler);
        res = RootSort(res1,res2,h1,h2,h3,h4,toler);
        return res;
     }
  }
  else
  { 
    double sb,sc,mb,mc,yb,yc,yd,hdelta,ry,hry[3];
    hb = hb/ha;    hc = hc/ha;    
    hd = hd/ha;    he = he/ha;
    ha = 1.0;
    double hdelta1, hdelta2;
	hdelta1 = hdelta2 = 1.0;
	ry = 1.0;
	
    yb = -hc; 
    yc = hb*hd - 4*he;
    yd = he*(4*hc - hb*hb) - hd*hd;
    res = Equat3(1,yb,yc,yd,hry[0],hry[1],hry[2],toler);
    int k = 0 ;
    for(k = 0; k < res; k++ )
    { 
        hdelta1 = 4*hry[k] + hb*hb - 4*hc;
        hdelta2 = hry[k]*hry[k] - 4*he;

        ry = hry[k];
        if (fabs(hdelta1) < toler && fabs(hdelta2) < toler)
        {
            hdelta1 = hdelta2 = 0.0;
            break;
        }
        else if (fabs(hdelta1) < toler && hdelta2 >= toler)
        {
            hdelta1 = 0.0;
            hdelta2 = sqrt(hdelta2);
            break;
        }
        else if (hdelta1 >= toler && fabs(hdelta2) < toler)
        {
            hdelta1 = sqrt(hdelta1);
            hdelta2 = 0.0;
            break;
        }
        else if (hdelta1 >= toler && hdelta2 >= toler)
        {
            hdelta1 = sqrt(hdelta1);
            hdelta2 = sqrt(hdelta2);
            break;
        }
    }
    if( k == res )
        return 0;

    hdelta = hb*ry - 2*hd;
    if (hdelta > -toler)
    {
        sb = (hb + hdelta1) * 0.5; 
        sc = (ry + hdelta2) * 0.5;
        mb = (hb - hdelta1) * 0.5; 
        mc = (ry - hdelta2) * 0.5;
    }
    else
    {
        sb = (hb + hdelta1) * 0.5; 
        sc = (ry - hdelta2) * 0.5;
        mb = (hb - hdelta1) * 0.5; 
        mc = (ry + hdelta2) * 0.5;
    }
    res1 = Equat2(1,sb,sc,h1,h2,toler);
    res2 = Equat2(1,mb,mc,h3,h4,toler);
    res = RootSort(res1,res2,h1,h2,h3,h4,toler);
    return res;     
  }     
}




//===============================================================================
//4. 根排序和剔除重根 , 用于Equat4();
int RootSort(int r1, int r2, double &rs1, double &rs2, double &rs3, double &rs4, double toler) 
{  
	int i, j, k = 0, m;
	double h[4], t[4];

	if (r1 > 0) h[k++] = rs1;
	if (r1 > 1) h[k++] = rs2;
	if (r2 > 0) h[k++] = rs3;
	if (r2 > 1) h[k++] = rs4;

	if (k == 0) return 0;
	
	for (i = m = 0; i < k; i++)
	{
		for (j = 0; j < m; j++)
			if (fabs(h[i] - t[j]) < toler) break;
		if (j == m)
			t[m++] = h[i];
	}

	rs1 = t[0];
	rs2 = t[1];
	rs3 = t[2];
	rs4 = t[3];
	
	return m;
/*
	if( (r1 == 0) && (r2 == 0) ) 
       return 0;
    else if( (r1 == 1) && (r2 == 0) )
       return 1;
    else if( (r1 == 2) && (r2 == 0) )
       return 2;
    else if( (r1 == 0) && (r2 == 1) ) 
    {  
       rs1 = rs3;      
       return 1;   
    }
    else if( (r1 == 1) && (r2 == 1) ) 
    {  
       if( fabs(rs3 - rs1) > toler )
       {
          rs2 = rs3;   
          return 2; 
       }
       else 
          return 1;  
    }   
    else if( (r1 == 2) && (r2 == 1) ) 
    {  
       if( (fabs(rs3 - rs1) > toler) && (fabs(rs3 - rs2) > toler) )
          return 3;  
       else 
          return 2;
    }
    else if( (r1 == 0) && (r2 == 2) ) 
    {
       rs1 = rs3;      rs2 = rs4;
       return 2;
    }   
    else if( (r1 == 1) && (r2 == 2) )
    {  
       if( fabs(rs3 - rs1) > toler ) 
       {
          if( fabs(rs4 - rs1) > toler )
          {
             rs2 = rs3;    rs3 = rs4; 
             return 3; 
          } 
          else 
          {
             rs2 = rs3;
             return 2;
          }  
       }
       else 
       {  
          if( fabs(rs4 - rs1) > toler )
          {
             rs2 = rs4;
             return 2; 
          } 
          else 
             return 1;
       }
    }
    else   //if( (r1 == 2) && (r2 == 2) )
    {
       if( (fabs(rs3 - rs1) > toler) && (fabs(rs3 - rs2) > toler) )
       {
          if( (fabs(rs4 - rs1) > toler) && (fabs(rs4 - rs2) > toler) )  
              return 4; 
          else 
              return 3;
       }
       else 
       {
          if( (fabs(rs4 - rs1) > toler) && (fabs(rs4 - rs2) > toler) )  
          {
             rs3 = rs4;     
             return 3; 
          }
          else
             return 3;   
       }
    }
*/
}  



//===============================================================================
//5. 返回高次方程的实根

int Equatn(int degree, double coef[], double root[], double toler)
{
    
    int  i,k,new_degree,root_num = 0,zero_root = 0, flag = 0;
    double en_r,en_p,en_q,new_coef[DEGREE_NUM], solu = 0.0; 
    
    // 找到第一个非零项，降阶
    for( k = 0; k <= degree - 4; k++ )
    {
        if( fabs(coef[k]) > toler )
            break;
    }       
    if( k > 0 )
    {
        for(i = 0; i <= degree -k; i++)
            coef[i] = coef[i+k];
        degree = degree - k; 
    }
    
    // 寻找倒数第一个非零项，降阶
    for( k = degree; k >= 4; k-- )
    {
        if( fabs(coef[k]) > toler )
            break;
    }       
    if( k < degree )   // 有零根
    {
        degree = k; 
        root[root_num] = 0;
        root_num++;
        zero_root = 1;
    }

    // 判别 a1 = a3  = a5 = ... = 0 的奇异情况
    if( (degree > 4) && (degree%2 == 0) )  
    {
        double rt[DEGREE_NUM];
        for( k = 1; k <= degree; k = k +2 )
        {
            if( fabs(coef[k]) > toler )
                break;
        }       
        if( k > degree )
        {
            for(i = 0; i <= degree/2; i++)
                new_coef[i] = coef[2*i];
            new_degree = degree/2; 
            if(new_degree == 4)
                flag = Equat4(new_coef[0], new_coef[1], new_coef[2], new_coef[3], new_coef[4], 
                              rt[0], rt[1], rt[2], rt[3], toler); 
            else if(new_degree == 3)
                flag = Equat3(new_coef[0], new_coef[1], new_coef[2], new_coef[3], 
                              rt[0], rt[1], rt[2], toler); 
            for( i = 0; i < flag; i++ )
            {
                if( rt[i] > 0 )
                {
                    root[root_num] = sqrt(rt[i]); 
                    root[root_num+1] = -1.0*sqrt(rt[i]); 
                    root_num = root_num +2;
                }
            } 
            return root_num; 
        }
    }

    // 将coef[]规格化
    for(i = 1; i <= degree; i++)
        coef[i] = coef[i]/coef[0];   
    coef[0] = 1.0;
    
    // 将coef[]赋给动态系数组new_coef[]
    for(i = 0; i < DEGREE_NUM; i++)
        if( i <= degree)
            new_coef[i] = coef[i];
        else 
            new_coef[i] = 0; 
    new_degree = degree;
                            
    // 方程系数奇异
    if( fabs(new_coef[new_degree]) < 0.01) 
    {   
        double initial;
        int regular = 0;
		for( i = 0; i <= 50; i++ )
		{
			for( k = 0; k < 2; k++ )
			{
			    initial = i/10;
			    if(k == 1)  
			    	initial = -initial;
			    regular = Newton(new_degree,new_coef,initial,solu);
			    if( regular == 1 )           
			    	break;
		    } 
		    if( regular == 1) 
		        break;
		}   
        if( i < 51 )  
        {
	        root[root_num] = solu;
	        reduce_single(new_degree,new_coef,solu); 
            root_num++;                
            new_degree--;
	        zero_root = 1;
	    }	
        else 
            return 0;
    }

    // 基于Bernouli迭代、Newton迭代修正求根
    int regular;
    double  fx,dfx;   
    while( new_degree > 4 ) 
    {   
        flag = Bernouli(new_degree,new_coef,en_r,en_p,en_q);
        if( flag == 1 )
        {   
            NewtonAided(degree,coef,en_r,fx,dfx);
            if( fabs(dfx) < toler )
            {
                root[root_num] = en_r; 
                reduce_single(new_degree,new_coef,en_r); 
                root_num++;
                new_degree -= 1;   
            }
            else
            {        
                regular = Newton(degree,coef,en_r,solu);           
                if( regular == 1 )
                {
                    root[root_num] = solu; 
                    reduce_single(new_degree,new_coef,solu); 
                    root_num++;
                    new_degree -= 1;   
                }    
                else                     
                {
                    root[root_num] = en_r; 
                    reduce_single(new_degree,new_coef,en_r); 
                    root_num++;
                    new_degree -= 1;   
                    // 1997-6-24 Modify for yj: circle-torus intersection
                }
            }        
        }
        else if( flag == 2 )
        {  
            if( en_p*en_p - 4*en_q > -toler ) 
            {
                double t1,t2;
                Equat2(1,en_p,en_q,t1,t2); 
                root[root_num] = t1; 
                root[root_num+1] = t2;  
                root_num = root_num + 2;
            }
            reduce_twice(new_degree,new_coef,en_p,en_q);
            new_degree -= 2;   
        }  
        else   // Bernouli法失效，Newton法作解搜索
        { 
            double  initial;
            regular = 0;
            for( i = 0; i <= 40; i++ )
            {
                initial = 4 - 8.0*i/40;
                regular = Newton(new_degree,new_coef,initial,solu);
                if( regular == 1 )           
                    break;
            }
            if( regular == 1 ) 
            {
                root[root_num] = solu; 
                reduce_single(new_degree,new_coef,solu); 
                root_num++;
                new_degree -= 1;   
            } 
            else  
                break;
        }
    }               
    
    if( new_degree <= 4 )
    {
        int res = 0;
        double h[4];
        if( new_degree == 4 )
            res = Equat4(new_coef[0],new_coef[1],new_coef[2],new_coef[3],new_coef[4],h[0],h[1],h[2],h[3],toler);  
        else if(new_degree == 3 ) 
            res = Equat3(new_coef[0],new_coef[1],new_coef[2],new_coef[3],h[0],h[1],h[2],toler); 
        else if(new_degree == 2 ) 
            res = Equat2(new_coef[0],new_coef[1],new_coef[2],h[0],h[1],toler); 
        else if(new_degree == 1)    
            res = Equat2(0,new_coef[0],new_coef[1],h[0],h[1],toler); 
        else 
            res = 0; 
        for( i = 0; i < res; i++ )
            root[root_num+i] = h[i];
        root_num += res;
    }
    
    if( degree > 4 )         // 高于4次的方程，可能因消去二次项而丢失精度
    {   
        for( i = zero_root; i < root_num; i++ )
        {   
            NewtonAided(degree,coef,root[i],fx,dfx);
            if( fabs(dfx) > toler )
            { 
                flag = Newton(degree,coef,root[i],solu);           
                if(flag == 1) 
                    root[i] = solu;  // 由于new_coef不精确，这几个根应用Newton法修正
            }
        } 
        if( root_num > 1 )   // 剔除重根
            Sort_reduce(root_num,root);
    } 
    return root_num; 
}

int EquatnNew(int degree, double coef[], double root[],double toler,double coefTol)
{
	/*
	//高次系数归一化
	if(fabs(coef[0]) > coefTol)
	{
		for(int kk = 0; kk <= degree;kk++ )
		{
			coef[kk] /= coef[0];
		}
	}
	else  //按低次方程求解
	{
		for(int kk = 0; kk < degree;kk++ )
		{
			coef[kk] = coef[kk+1];
		}
		degree--;
		return EquatnNew(degree,coef,root,toler,coefTol);
	}
	*/
	//-------to remove warning----2004-07-23--------
	coefTol = 1e-20;
	//------------------------------------
	double low = -1;
	double upper = 1;
	double *root1,*root2;
	double *coef2;
	int num1,num2,num = 0;
  
	root1= new double[degree];
	root2= new double[degree];
	coef2 = new double[degree+1];

	num1 = Equatn(degree, coef, root1, low,upper,toler);
	//作变换，modified by lxz:2003-11-17
	for(int ii=0; ii< degree+1;ii++)
        coef2[ii] = coef[degree-ii];
	num2 = Equatn(degree, coef2, root2, low,upper,toler);
	for(int ii = 0;ii< num1;ii++)
	{
		root[ii] = root1[ii];
		num = num1;
	}
	for(int ii= 0;ii< num2;ii++)
	{
		if(fabs(root2[ii]) > toler && fabs(root2[ii]-1) > toler &&fabs(root2[ii]+1) > toler)
		{
			root[num] =  1/root2[ii];
			num++;
		}
	}

	delete []root1;
	delete []root2;
	delete []coef2;

	return num;
}

int EquatnNew(int degree, double coef[], double root[],int& step,double toler)
{
	double low = -1;
	double upper = 1;
	double *root1,*root2;
	double *coef2;
	int num1,num2,num = 0;
	step = 0;
	root1= new double[degree];
	root2= new double[degree];
	coef2 = new double[degree+1];
	int steps = 0;
	num1 = Equatn(degree, coef, root1, low,upper,steps,toler);
	//作变换，modified by lxz:2003-11-17
	for(int ii=0; ii< degree+1;ii++)
        coef2[ii] = coef[degree-ii];
	step += steps;
	steps = 0;
	num2 = Equatn(degree, coef2, root2, low,upper,steps,toler);
	step += steps;
	for(int ii = 0;ii< num1;ii++)
	{
		root[ii] = root1[ii];
		num = num1;
	}
	for(int ii= 0;ii< num2;ii++)
	{
		if(fabs(root2[ii]) > toler && fabs(root2[ii]-1) > toler &&fabs(root2[ii]+1) > toler)
		{
			root[num] =  1/root2[ii];
			num++;
		}
	}
	
	delete []root1;
	delete []root2;
	delete []coef2;
	
	return num;
}

//利用求导的思想解高次一元方程,解4次以上(不含4次)的方程，系数为从高次到低次
int Equatn(int degree, double coef[], double root[],double low,double upper,double toler,double coefTol)
{

	if(degree >= 4 )
	{
		double *d0Coef;  //存放求导后的次数
		double *d1Coef;
		double *dRoot;  //存放求导后解的个数，最多degree个！
		double *tmpRoot;
		int num;        
		int tmpNum;
		d0Coef = new double[degree+1]; 
		d1Coef = new double[degree+1];
		dRoot = new double[degree+2];
		tmpRoot = new double[degree];

		int i,j;
		int tmpDegree = degree;
	
		for(i = 0 ;i < degree+1;i++)
		{
			d0Coef[i] = coef[i];
		}

			/*
		//高次系数归一化
		if(fabs(d0Coef[0]) > coefTol)
		{
			for(int kk = 0; kk <= degree;kk++ )
			{
				d0Coef[kk] /= d0Coef[0];
			}
		}
		else  //按低次方程求解
		{
			for(int kk = 0; kk < degree;kk++ )
			{
				d0Coef[kk] = d0Coef[kk+1];
			}
			degree--;
			return Equatn(degree,coef,root,low,upper,toler,coefTol);
		}
		*/
	/////////////////////////////////
	//求导到4次
		for(i = 0; i < degree -4; i++)
		{
			for(j = 0;j < tmpDegree;j++ )
			{
				d1Coef[j] = (tmpDegree-j)*d0Coef[j]; 
			}
        
			for(j = 0 ;j < tmpDegree;j++)
			{
				d0Coef[j] = d1Coef[j];
			} 
			tmpDegree--;
		}
   
		double h1,h2,h3,h4;
		tmpNum = Equat4(d1Coef[0],d1Coef[1],d1Coef[2],d1Coef[3],d1Coef[4],h1,h2,h3,h4,toler);
		//排序
		tmpRoot[0] = h1;
		tmpRoot[1] = h2;
		tmpRoot[2] = h3;
		tmpRoot[3] = h4;
		double tmp;
		for(i= 0;i< tmpNum-1;i++)
		for(j = i+1;j<tmpNum;j++)
		{
			if(tmpRoot[j] < tmpRoot[i])
			{
				tmp = tmpRoot[j];
				tmpRoot[j] = tmpRoot[i];
				tmpRoot[i] = tmp;
			}
		}

		bool hasLow;
		bool hasUp;
		int curNum = 0;

		for(int k = 5; k <= degree; k++ )
		{
			///得到区间;
			num = 0;
			hasLow = false;
			hasUp = false;
			for(i = 0; i < tmpNum;i++)
			{
				if(fabs(tmpRoot[i]-low) < toler)
				{
					dRoot[num] = low;
					num++;
				}
				else if(fabs(tmpRoot[i]-upper) < TOLER)
				{
					dRoot[num] = upper;
					num++;
				}
				else if(tmpRoot[i] > low && tmpRoot[i] < upper)
				{
					dRoot[num] = tmpRoot[i];
					num++;
				}
			}
		
			for(i = num; i > 0;i--)
				dRoot[i] = dRoot[i-1];
			dRoot[0] = low;
			num++;
		
			dRoot[num] = upper;
			num++;
	
			//计算高一次方程的解
			//高一次方程的系数;
			tmpDegree = degree;
			for(i = 0 ;i < degree+1;i++)
			{
				d0Coef[i] = coef[i];
			}
			for(i = 0; i < degree - k; i++)  //  5 
			{
				for(j = 0;j < tmpDegree;j++ )
				{
					d1Coef[j] = (tmpDegree-j)*d0Coef[j]; 
				}
			
				for(j = 0 ;j < tmpDegree+1;j++)
				{
					d0Coef[j] = d1Coef[j];
				} 
				tmpDegree--;
			}
			/////////////////////////////////////////////
			//对每一个区间判断是否有解,有解的求解
			double val;
			for(i = 0;i < num-1;i++)
			{
				//二分法计算根
				if(binarySolution(k,d0Coef,dRoot[i],dRoot[i+1],val,toler))	
				{
					tmpRoot[curNum] = val;
					curNum ++;
				}
			}
			//////////////////////////
			//设置循坏参数，此时排序
			tmpNum = curNum;
			curNum = 0;
	   
		   for(i= 0;i< tmpNum-1;i++)
			   for(j = i+1;j<tmpNum;j++)
			   {
				   if(tmpRoot[j] < tmpRoot[i])
					{
					    tmp = tmpRoot[j];
						tmpRoot[j] = tmpRoot[i];
						tmpRoot[i] = tmp;
					}
				}
		 }
		///////////////////////////////////////
		for(i = 0;i < tmpNum;i++)
			root[i] = tmpRoot[i];
		delete []d0Coef;
		delete []d1Coef;
		delete []dRoot;
		delete []tmpRoot;

		return tmpNum;
	}
	else if(degree == 3)
	{
		double h1,h2,h3;
		int num = 0;
		double tmpRoot[3];
		int tmpNum = Equat3(coef[0], coef[1], coef[2], coef[3],h1,h2,h3,toler);
		tmpRoot[0] = h1;
		tmpRoot[1] = h2;
		tmpRoot[2] = h3;
		for(int i = 0; i < tmpNum;i++ )
		{
			if(fabs(tmpRoot[i]-low) < toler || fabs(tmpRoot[i]-upper) < toler ||
				(tmpRoot[i] > low && tmpRoot[i] < upper))
			{
				root[num] = tmpRoot[i];
				num++;
			}
		}
		return num;
	}
	else if(degree == 2)
	{
		double h1,h2;
		int num = 0;
		double tmpRoot[2];
		int tmpNum = Equat2(coef[0], coef[1], coef[2],h1,h2,toler);
		tmpRoot[0] = h1;
		tmpRoot[1] = h2;
		
		for(int i = 0; i < tmpNum;i++ )
		{
			if(fabs(tmpRoot[i]-low) < toler || fabs(tmpRoot[i]-upper) < toler ||
				(tmpRoot[i] > low && tmpRoot[i] < upper))
			{
				root[num] = tmpRoot[i];
				num++;
			}
		}
		return num;
	}
	else if(degree == 1)
	{
		double tmpRoot;
		int num = 0;
		if(fabs(coef[0]) >coefTol)
		{
			tmpRoot = -coef[1]/coef[0];
			if(fabs(tmpRoot-low) < toler || fabs(tmpRoot-upper) < toler ||
				(tmpRoot > low && tmpRoot < upper))
			{
				root[num] = tmpRoot;
				num++;
			}
		}
		return num;
	}
	else 
	{
		return 0;
	}

	
	
}

int Equatn(int degree, double coef[], double root[],double low,double upper,int &step,double toler)
{
	double *d0Coef;  //存放求导后的次数
	double *d1Coef;
	double *dRoot;  //存放求导后解的个数，最多degree个！
	double *tmpRoot;
	int num;        
	int tmpNum;
	d0Coef = new double[degree+1]; 
	d1Coef = new double[degree+1];
	dRoot = new double[degree+2];
	tmpRoot = new double[degree];
    int i,j;
    int tmpDegree = degree;
	
	for(i = 0 ;i < degree+1;i++)
	{
		d0Coef[i] = coef[i];
	}
/////////////////////////////////
//求导到4次
	for(i = 0; i < degree -4; i++)
	{
		for(j = 0;j < tmpDegree;j++ )
		{
			d1Coef[j] = (tmpDegree-j)*d0Coef[j]; 
		}
        
		for(j = 0 ;j < tmpDegree;j++)
		{
			d0Coef[j] = d1Coef[j];
		} 
		tmpDegree--;
	}
   
   double h1,h2,h3,h4;
   tmpNum = Equat4(d1Coef[0],d1Coef[1],d1Coef[2],d1Coef[3],d1Coef[4],h1,h2,h3,h4,toler);
   //排序
   tmpRoot[0] = h1;
   tmpRoot[1] = h2;
   tmpRoot[2] = h3;
   tmpRoot[3] = h4;
   double tmp;
   for(i= 0;i< tmpNum-1;i++)
   for(j = i+1;j<tmpNum;j++)
   {
	   if(tmpRoot[j] < tmpRoot[i])
	   {
		   tmp = tmpRoot[j];
		   tmpRoot[j] = tmpRoot[i];
		   tmpRoot[i] = tmp;
	   }
   }

//   bool hasLow;
//   bool hasUp;
   int curNum = 0;
   step = 0;
   
   for(int k = 5; k <= degree; k++ )
   {
        ///得到区间;
		num = 0;
		
        //计算高一次方程的解
		//高一次方程的系数;
		tmpDegree = degree;
		for(i = 0 ;i < degree+1;i++)
		{
			d0Coef[i] = coef[i];
		}
		for(i = 0; i < degree - k; i++)  //  5 
		{
			for(j = 0;j < tmpDegree;j++ )
			{
				d1Coef[j] = (tmpDegree-j)*d0Coef[j]; 
			}
			
			for(j = 0 ;j < tmpDegree+1;j++)
			{
				d0Coef[j] = d1Coef[j];
			} 
			tmpDegree--;
		}
		/////////////////////////////////////
		for(i = 0; i < tmpNum;i++)
		{
			if(fabs(tmpRoot[i]-low) < toler)
			{
				dRoot[num] = low;
				num++;
			}
			else if(fabs(tmpRoot[i]-upper) < TOLER)
			{
				dRoot[num] = upper;
				num++;
			}
			else if(tmpRoot[i] > low && tmpRoot[i] < upper)
			{
				dRoot[num] = tmpRoot[i];
				num++;
			}
		}
		
		for(i = num; i > 0;i--)
		    dRoot[i] = dRoot[i-1];
		dRoot[0] = low;
		num++;
		
		dRoot[num] = upper;
		num++;
		
		
        /////////////////////////////////////////////
		//对每一个区间判断是否有解,有解的求解
		double val;
		int steps = 0;
		for(i = 0;i < num-1;i++)
		{
			//二分法计算根
			if(binarySolution(k,d0Coef,dRoot[i],dRoot[i+1],val,steps,toler))	
			{
				tmpRoot[curNum] = val;
				curNum ++;
				step += steps;
			}
			
		}
        //////////////////////////
		//设置循坏参数，此时排序
		tmpNum = curNum;
		curNum = 0;

		for(i= 0;i< tmpNum-1;i++)
			for(j = i+1;j<tmpNum;j++)
			{
				if(tmpRoot[j] < tmpRoot[i])
				{
					tmp = tmpRoot[j];
					tmpRoot[j] = tmpRoot[i];
					tmpRoot[i] = tmp;
				}
			}
			
     }
///////////////////////////////////////
    for(i = 0;i < tmpNum;i++)
		root[i] = tmpRoot[i];
	delete []d0Coef;
	delete []d1Coef;
	delete []dRoot;
	delete []tmpRoot;
	
	return tmpNum;
}

bool  binarySolution(int degree,double coef[],double left,double right,double & val,double toler,double coefTol)
{
    /*
	//高次系数归一化
	if(fabs(coef[0]) > coefTol)
	{
		for(int kk = 0; kk <= degree;kk++ )
		{
			coef[kk] /= coef[0];
		}
	}
	else  //按低次方程求解
	{
		for(int kk = 0; kk < degree;kk++ )
		{
			coef[kk] = coef[kk+1];
		}
		degree--;
		return binarySolution(degree,coef,left,right,val,toler,coefTol);
	}
    */
	//-------to remove warning----2004-07-23--------
	coefTol = 1e-20;
	//------------------------------------

	double mid,mleft,mright;
	double oldMid;
	double lValue,rValue,mValue;
	mleft = left;
	mright = right;
	lValue = coef[0]*mleft+coef[1];
	rValue = coef[0]*mright+coef[1];
	for(int ii = 1;ii < degree;ii++)
	{
		lValue *= mleft;
		lValue += coef[ii+1];	
		rValue *= mright;
		rValue += coef[ii+1];	
	}
	if(fabs(lValue) < toler)
	{
		val = left;
		return true;
	}
	else if(fabs(rValue) < toler)
	{
		val = right;
		return true;
	}
	if(lValue* rValue < 0)  //二分法计算根
	{
		mValue = lValue;
		int steps = 0;
		mid = (mright+mleft)/2;
		oldMid = left;
		while(fabs(oldMid-mid) > toler) 
		{
			mValue = coef[0]*mid+coef[1];
			for(int ii = 1;ii < degree;ii++)
			{
				mValue *= mid;
				mValue += coef[ii+1];	
			}
			if(lValue*mValue < 0)
			{
				mright = mid;
			}
			else
			{
				mleft = mid;
			}
			steps ++;
			if(steps > 200)
			{
				return false;
			}
			oldMid = mid;
			mid = (mright+mleft)/2;
		}
		val = mid;
		return true;
		
	}
	else if(lValue* rValue > 0 )
	{
		return false;
	}
	return false;
}

bool  binarySolution(int degree,double coef[],double left,double right,double &val,int &step,double toler)
{
	double mid,mleft,mright;
	double oldMid;
	double lValue,rValue,mValue;
	mleft = left;
	mright = right;
	lValue = coef[0]*mleft+coef[1];
	rValue = coef[0]*mright+coef[1];
	step = 0;
	for(int ii = 1;ii < degree;ii++)
	{
		lValue *= mleft;
		lValue += coef[ii+1];	
		rValue *= mright;
		rValue += coef[ii+1];	
	}
	if(fabs(lValue) < toler)
	{
		val = left;
		return true;
	}
	else if(fabs(rValue) < toler)
	{
		val = right;
		return true;
	}
	if(lValue* rValue < 0)  //二分法计算根
	{
		mValue = lValue;
		int steps = 0;
		mid = (mright+mleft)/2;
		oldMid = left;
		while(fabs(oldMid-mid) > toler) 
		{
			mValue = coef[0]*mid+coef[1];
			for(int ii = 1;ii < degree;ii++)
			{
				mValue *= mid;
				mValue += coef[ii+1];	
			}
			if(lValue*mValue < 0)
			{
				mright = mid;
			}
			else
			{
				mleft = mid;
			}
			steps ++;
			if(steps > 200)
			{
				return false;
			}
			oldMid = mid;
			mid = (mright+mleft)/2;
		}
		val = mid;
		step = steps;
		return true;
		
	}
	else if(lValue* rValue > 0 )
	{
		return false;
	}

	return false;
}


//===============================================================================
// 6. 多项式消去因子（x-fa), degree 为多项式的次数，coef为系数
void reduce_single(int degree, double coef[], double fa)
{
    for( int i = 1; i<= degree - 1; i++ )
        coef[i] = coef[i] + fa*coef[i-1];
}


//===============================================================================
// 7. 多项式消去因子（x^2+fa*x+fb), degree 为多项式的次数，coef为系数
void reduce_twice(int degree, double coef[], double fa, double fb)
{
    coef[1] = coef[1] - fa*coef[0];
    for( int i = 2; i<= degree - 2; i++ )
        coef[i] = coef[i] - fa*coef[i-1] - fb*coef[i-2];
}   



//===============================================================================
// 8. 剔除重复者
void Sort_reduce(int &rn, double rt[])
{   
    int flag,i,j,numb = 1;
    double srt[DEGREE_NUM];

    for( i = 0; i <= rn-1; i++ )
        srt[i] = rt[i];

    for( i = 1; i <= rn-1; i++ )
    {        
        flag = 0;
        for( j = 0; j<= numb -1; j++ )
            if(fabs(srt[i] - rt[j]) < TOLER )
            {   flag = 1;   break; }
        if( flag == 0 )         
        {
            rt[numb] = srt[i];
            numb++; 
        }
    }           
    rn = numb;
}




//===============================================================================
// 9. Newton法求代数方程的根： 
// 次数 degree, 系数：coef, 初值：start_value, 传引用返回根 root
int Newton(int degree,double coef[],double start_value,double &root)
{
    double x,nextx,fx,dfx;
    int recu_num = 0;
    nextx = start_value;
    do
    {   
        x = nextx;
        NewtonAided(degree,coef,x,fx,dfx);
        if( fabs(fx) < 1e-15 )       
        {
            root = x;
            return 1;
        }            
        if( fabs(dfx) < TOLER*0.001 )
        {   
            if( fabs(fx) < TOLER*0.001 ) 
            {
                root = x;
                return 1;
            }            
            else 
                return 0;
        }                   
        nextx = x - fx/dfx;          
        recu_num++;    
    }while( (fabs(x - nextx) > TOLER*0.001) && (recu_num < ITERAT_NUM) );
    if( recu_num < ITERAT_NUM ) 
    {
        root = nextx; 
        return 1;
    }
    else
        return 0;
}

 

//===============================================================================
// 10. 用于Newton()                                   
void NewtonAided(int num, double a[], double x, double &fx, double &dfx)
// recursive computing f(x_k) and f'(x_k) for x_{k+1} = x_k - f(x_k)/f'(x_k)
{
    double b[DEGREE_NUM],c[DEGREE_NUM];
    b[0] = a[0];   
    c[0] = b[0];
    for( int i = 1; i<= num -1; i++ )
    {
        b[i] = a[i] + x*b[i-1];
        c[i] = b[i] + x*c[i-1];
    }
    dfx = c[num-1];   
    fx = a[num] + x*b[num-1];
}



//===========================================================
// 11-1  Bernouli第一迭代
int Bernouli_Iterate_S(int degree, double coefficent[], double &root, double &p, double &q)
{
    int i,j,res = 0;
    double s[ITERAT_NUM],next_root,next_p,next_q,temp;
        
    root = 9.87654321;
    p = 0.0;            q = 0; 
        
    s[0] = 1.0;
    s[1] = -  coefficent[1];
    s[2] = - (coefficent[1]*s[1] + 2*coefficent[2]);
      
    for( i = 3; i <= ITERAT_NUM - 1; i++ )  
    {
        if( i <= degree )
        {               
            s[i] = 0;
            for( j = 1; j <= i-1; j++ )
                s[i] += coefficent[j]*s[i-j]; 
            s[i] += i*coefficent[i]; s[i] = 0.0 - s[i];
        }
        else
        {
            s[i] = 0;
            for( j = 1; j <= degree; j++ )
                s[i] += coefficent[j]*s[i-j]; 
            s[i] = 0.0 - s[i];
        } 
           
        if( fabs(s[i-1]) > TOLER ) 
            next_root = s[i]/s[i-1];
        else 
            next_root = 1.23456789;
        if( fabs(next_root - root) < 1.0e-8 )  // 1.0e-5;
        {
            res = 1; 
            break;
        }                         
            
        temp = s[i-2]*s[i-2] - s[i-1]*s[i-3];
        if( fabs(temp) > TOLER ) 
        {
            next_p = - (s[i-1]*s[i-2] - s[i]*s[i-3])/temp;
            next_q = (s[i-1]*s[i-1] - s[i]*s[i-2])/temp; 
        }
        else 
        {   
            next_p = 1.23456789;
            next_q = 9.87654321;
        }
        if( ( (fabs(next_p - p) < TOLER*0.001) && (fabs(next_q - q) < TOLER*0.001) ) || 
            ( (fabs(next_p - p) < 1.0e-4) && (fabs(next_q - q) < 1.0e-4) && 
              ( i >= ITERAT_NUM - 2 ) ) )
        {
            if( i > 32 )        
            {
                res = 2;
                break;
            }
        }    
            
        root = next_root;   p = next_p;   q = next_q;
    }   
    return res;
}


//===========================================================
// 11-2 Bernouli第二迭代
int Bernouli_Iterate_L(int degree, double coefficent[], double &root, double &p, double &q)
{
    
    int   i,j,res = 0;//,flag = 0
    double xi[ITERAT_NUM][DEGREE_NUM],lambda[ITERAT_NUM],next_root,next_p,next_q,temp;
        
    lambda[1] = -1.0/coefficent[1]; 
    xi[2][1] = coefficent[1] - 2.0*coefficent[2]/coefficent[1];
    lambda[2] = -1.0/xi[2][1]; 
    root = 1.0/lambda[2];
    p = 0;    q = 0;           
        
    for( i = 3; i <= ITERAT_NUM - 1; i++ )  
    {
        if( i <= degree )
        {
            xi[i][1] = coefficent[i-1] - i*coefficent[i]/coefficent[1];
            for( j = 2; j <= i-1; j++ )
                xi[i][j] = xi[i][j-1]*lambda[j] + coefficent[i-j];
            if( fabs(xi[i][i-1]) > TOLER )
                lambda[i] = -1.0/xi[i][i-1];  
            else
                return -1;
        }
        else
        {
            xi[i][1] = coefficent[degree -1] + coefficent[degree]*lambda[i-degree+1]; 
            for( j = 2; j <= degree - 1; j++ )
                xi[i][j] = coefficent[degree-j] +  xi[i][j-1]*lambda[i-degree+j];
            if( fabs(xi[i][degree-1]) > TOLER )
                lambda[i] = -1.0/xi[i][degree-1];
            else 
                return -1;  
        } 
            
           
        if( fabs(lambda[i]) > TOLER ) 
            next_root = 1.0/lambda[i];     
        else
            next_root = 1.23456789;
        if( fabs(next_root - root) < 1.0e-8 )  // 1.0e-5
        {
                res = 1; 
                break; 
        }
        
        temp = lambda[i]*(lambda[i-1] - lambda[i-2]);
        if( fabs(temp) > TOLER ) 
        {
            next_p = - (lambda[i] - lambda[i-2])/temp;
            next_q = (lambda[i] - lambda[i-1])/(lambda[i-1]*temp); 
        }
        else 
        {   
            next_p = 1.23456789;
            next_q = 9.87654321;
        }
        
        if( ( (fabs(next_p - p) < TOLER*0.001) && (fabs(next_q - q) < TOLER*0.001) ) || 
            ( (fabs(next_p - p) < 1.0e-4) && (fabs(next_q - q) < 1.0e-4) && 
              ( i >= ITERAT_NUM - 2 ) ) )
        {
            if( i > 32 )        
            {
                res = 2;
                break;
            }
        } 
        root = next_root;   p = next_p;   q = next_q;  
    }          
    return res;
}

                                   
    
//===============================================================================
// 11. Bernouli法求代数方程的根： 
// 次数 degree, 系数：coef, 传引用返回根 root
int Bernouli(int degree, double coefficent[], double &root, double &p, double &q)
{
    int i;//,resu = 0;
    double coef1[ITERAT_NUM],coef2[ITERAT_NUM];
    
    for(i = 1; i<= degree; i++)
    {
        coef1[i] = coefficent[i]/coefficent[0];  
        coef2[i] = coef1[i]; 
    }
    coef1[0] = 1;      coef2[0] = 1;
    
    if( (fabs(coef1[1]) < TOLER) || (fabs(coef1[1]*coef1[1] -2.0*coef1[2]) < TOLER) )
        return Bernouli_Iterate_S(degree, coef1, root, p, q);
    else
    {
        int resu = Bernouli_Iterate_L(degree, coef1, root, p, q);
        if( resu == -1 )
            return Bernouli_Iterate_S(degree, coef2, root, p, q);  
        else 
            return resu;    
    }
}
            
} // namespace icesop

