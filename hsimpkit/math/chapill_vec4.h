/******************************************************************************
ORIGINALLY Vec4
THE ORIGINAL Vec3 HAS CHANGED TO ChapillVec3
THE ORIGINAL Vec4 HAS CHANGED TO ChapillVec4
PLEASE REFER TO 'chapill_vec3'
	-HT
******************************************************************************/

//-----------------------------------------------------------------------------
// File : vec4f.hpp
//-----------------------------------------------------------------------------
// GLVU : Copyright 1997 - 2002 
//        The University of North Carolina at Chapel Hill
//-----------------------------------------------------------------------------
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without
// fee, provided that the above copyright notice appear in all copies
// and that both that copyright notice and this permission notice
// appear in supporting documentation.  Binaries may be compiled with
// this software without any royalties or restrictions.
//
// The University of North Carolina at Chapel Hill makes no representations 
// about the suitability of this software for any purpose. It is provided 
// "as is" without express or implied warranty.

//==========================================================================
// vec4.hpp : 4d vector class template. Works for any integer or real type.
//==========================================================================

#pragma once

#include <stdio.h>
#include <math.h>
namespace mymath{

template <class Type>
class ChapillVec4
{
  public:
    Type x, y, z, w;

    ChapillVec4 (void) 
      {};
    ChapillVec4 (const Type X, const Type Y, const Type Z, const Type W) 
      { x=X; y=Y; z=Z; w=W; };
    ChapillVec4 (const ChapillVec4& v) 
      { x=v.x; y=v.y; z=v.z; w=v.w; };
    ChapillVec4 (const Type v[4])
      { x=v[0]; y=v[1]; z=v[2]; w=v[3]; };
    void Set (const Type X, const Type Y, const Type Z, const Type W)
      { x=X; y=Y; z=Z; w=W; }
    void Set (const Type v[4])
      { x=v[0]; y=v[1]; z=v[2]; w=v[3]; };

    operator Type*()                             // Type * CONVERSION
      { return (Type *)&x; }
    operator const Type*() const                 // CONST Type * CONVERSION
      { return &x; }

    ChapillVec4& operator = (const ChapillVec4& A)            // ASSIGNMENT (=)
      { x=A.x; y=A.y; z=A.z; w=A.w; 
        return(*this);  };

    bool operator == (const ChapillVec4& A) const        // COMPARISON (==)
      { return (x==A.x && y==A.y && 
        z==A.z && w==A.w); }
    bool operator != (const ChapillVec4& A) const        // COMPARISON (!=)
      { return (x!=A.x || y!=A.y || 
        z!=A.z || w!=A.w); }

    ChapillVec4 operator + (const ChapillVec4& A) const       // ADDITION (+)
      { ChapillVec4 Sum(x+A.x, y+A.y, z+A.z, w+A.w); 
        return(Sum); };
    ChapillVec4 operator - (const ChapillVec4& A) const       // SUBTRACTION (-)
      { ChapillVec4 Diff(x-A.x, y-A.y, z-A.z, w-A.w);
        return(Diff); };
    Type operator * (const ChapillVec4& A) const       // DOT-PRODUCT (*)
      { Type DotProd = x*A.x+y*A.y+z*A.z+w*A.w; 
        return(DotProd); };
    ChapillVec4 operator * (const Type s) const        // MULTIPLY BY SCALAR (*)
      { ChapillVec4 Scaled(x*s, y*s, z*s, w*s); 
        return(Scaled); };
    ChapillVec4 operator / (const Type s) const        // DIVIDE BY SCALAR (/)
      { ChapillVec4 Scaled(x/s, y/s, z/s, w/s);
        return(Scaled); };
    ChapillVec4 operator & (const ChapillVec4& A) const       // COMPONENT MULTIPLY (&)
      { ChapillVec4 CompMult(x*A.x, y*A.y, z*A.z, w*A.w);
        return(CompMult); }

    friend inline ChapillVec4 operator *(Type s, const ChapillVec4& v)  // SCALAR MULT s*V
      { return ChapillVec4(v.x*s, v.y*s, v.z*s, v.w*s); }

    ChapillVec4& operator += (const ChapillVec4& A)      // ACCUMULATED VECTOR ADDITION (+=)
      { x+=A.x; y+=A.y; z+=A.z; w+=A.w; 
        return *this; }
    ChapillVec4& operator -= (const ChapillVec4& A)      // ACCUMULATED VECTOR SUBTRCT (-=)
      { x-=A.x; y-=A.y; z-=A.z; w-=A.w;
        return *this; }
    ChapillVec4& operator *= (const Type s)       // ACCUMULATED SCALAR MULT (*=)
      { x*=s; y*=s; z*=s; w*=s; 
        return *this; }
    ChapillVec4& operator /= (const Type s)       // ACCUMULATED SCALAR DIV (/=)
      { x/=s; y/=s; z/=s; w/=s;
        return *this; }
    ChapillVec4& operator &= (const ChapillVec4& A)  // ACCUMULATED COMPONENT MULTIPLY (&=)
      { x*=A.x; y*=A.y; z*=A.z; w*=A.w; return *this; }
    ChapillVec4 operator - (void) const          // NEGATION (-)
      { ChapillVec4 Negated(-x, -y, -z, -w);
        return(Negated); };

/*
    const Type& operator [] (const int i) const // ALLOWS VECTOR ACCESS AS AN ARRAY.
      { return( (i==0)?x:((i==1)?y:((i==2)?z:w)) ); };
    Type & operator [] (const int i)
      { return( (i==0)?x:((i==1)?y:((i==2)?z:w)) ); };
*/

    Type Length (void) const                     // LENGTH OF VECTOR
      { return ((Type)sqrt(x*x+y*y+z*z+w*w)); };
    Type LengthSqr (void) const                  // LENGTH OF VECTOR (SQUARED)
      { return (x*x+y*y+z*z+w*w); };
    ChapillVec4& Normalize (void)                       // NORMALIZE VECTOR
      { Type L = Length();                       // CALCULATE LENGTH
        if (L>0) { x/=L; y/=L; z/=L; w/=L; }
        return *this;
      };                                          // DIV COMPONENTS BY LENGTH

    void Wdiv(void)
      { x/=w; y/=w; z/=w; w=1; }

    void Print() const
      { printf("(%.3f, %.3f, %.3f, %.3f)\n",x, y, z, w); }

    static ChapillVec4 ZERO;
};

typedef ChapillVec4<float> ChapillVec4f;
typedef ChapillVec4<double> ChapillVec4d;

template<class Type> ChapillVec4<Type> ChapillVec4<Type>::ZERO = ChapillVec4<Type>(0,0,0,0);

}

