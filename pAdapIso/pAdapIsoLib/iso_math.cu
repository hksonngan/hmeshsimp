/*
 *  Math Routines For CUDA Kernels
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef _ISO_MATH_CUDA_INC_
#define _ISO_MATH_CUDA_INC_

template<typename T>
__device__ __forceinline__
void swapVal(T &a, T &b) {
	T x;
	x = a;
	a = b;
	b = x;
}

// Quadric Error Matrix in 5x5
template<typename T>
class QeMatrix5 {
public:
	T M00, M01, M02, M03, M04;
	T      M11, M12, M13, M14;
	T           M22, M23, M24;
	T                M33, M34;
	T                     M44;

	__device__ __forceinline__ QeMatrix5();
	__device__ __forceinline__ void addPlane(
		const T &a0, const T &a1, const T &a2, const T &a3, const T &a4);
	__device__ __forceinline__ void evaluateError(
		const T&a0, const T&a1, const T&a2, const T&a3, T &error);
};

template<typename T>
__device__ __forceinline__
QeMatrix5<T>::QeMatrix5() {
	M00 = M01 = M02 = M03 = M04 = 0;
	      M11 = M12 = M13 = M14 = 0;
	            M22 = M23 = M24 = 0;
	                  M33 = M34 = 0;
	                        M44 = 0;
}

template<typename T>
__device__ __forceinline__
void QeMatrix5<T>::addPlane(
	const T &a0, const T &a1, const T &a2, const T &a3, const T &a4
	) {
	M00 += a0*a0;
	M01 += a0*a1;
	M02 += a0*a2;
	M03 += a0*a3;
	M04 += a0*a4;

	M11 += a1*a1;
	M12 += a1*a2;
	M13 += a1*a3;
	M14 += a1*a4;

	M22 += a2*a2;
	M23 += a2*a3;
	M24 += a2*a4;

	M33 += a3*a3;
	M34 += a3*a4;

	M44 += a4*a4;
}

template<typename T>
__device__ __forceinline__
void QeMatrix5<T>::evaluateError(
	const T&a0, const T&a1, const T&a2, const T&a3, T &error
	) {
	error = M00*a0*a0 + 2*M01*a0*a1 + 2*M02*a0*a2 + 2*M03*a0*a3 + 2*M04*a0 +
		                  M11*a1*a1 + 2*M12*a1*a2 + 2*M13*a1*a3 + 2*M14*a1 +
						                M22*a2*a2 + 2*M23*a2*a3 + 2*M24*a2 +
										              M33*a3*a3 + 2*M34*a3 +
													                M44;
}

/* matrix solver for the linear system
   of the 4-dimensional hyperplanes */
template<typename T>
class MatrixSolver5 {
	typedef T* TP;
	const T tolerance;

public:
	// column major layout, elements 
	// in one row resides beneath
	T M[4][5];

	__device__ __forceinline__ MatrixSolver5();
	__device__ __forceinline__ void assign(QeMatrix5<T> qem);
	__device__ __forceinline__ bool solve(T &x, T &y, T &z, T &w);

private:
	__device__ __forceinline__ void swapRow(T r1[], T r2[]);
	__device__ __forceinline__ void divRow(T r[], const T d);
	__device__ __forceinline__ void subtractRow(T r1[], const T r2[]);
	__device__ __forceinline__ bool zero(const T &a);
};

template<typename T>
__device__ __forceinline__
MatrixSolver5<T>::MatrixSolver5(): tolerance(0.000001) {}
//	M[0][0] = M[0][1] = M[0][2] = M[0][3] = M[0][4] = 0;
//	M[1][0] = M[1][1] = M[1][2] = M[1][3] = M[1][4] = 0;
//	M[2][0] = M[2][1] = M[2][2] = M[2][3] = M[2][4] = 0;
//	M[3][0] = M[3][1] = M[3][2] = M[3][3] = M[3][4] = 0;
//}

template<typename T>
__device__ __forceinline__
void MatrixSolver5<T>::assign(QeMatrix5<T> qem) {
	M[0][0]           = qem.M00;
	M[0][1] = M[1][0] = qem.M01;
	M[0][2] = M[2][0] = qem.M02;
	M[0][3] = M[3][0] = qem.M03;
	M[0][4]           = -qem.M04;

	M[1][1]           = qem.M11;
	M[1][2] = M[2][1] = qem.M12;
	M[1][3] = M[3][1] = qem.M13;
	M[1][4]           = -qem.M14;

	M[2][2]           = qem.M22;
	M[2][3] = M[3][2] = qem.M23;
	M[2][4]           = -qem.M24;

	M[3][3]           = qem.M33;
	M[3][4]           = -qem.M34;
}

template<typename T>
__device__ __forceinline__
bool MatrixSolver5<T>::solve(T &x, T &y, T &z, T &w) {
	// gauss elimination
	for (char i = 0; i <= 2; i ++) {
		char j;
		if (zero(M[i][i])) {
			for (j = i + 1; j <= 3; j ++) {
				if (!zero(M[j][i]))
					break;
			}
			if (j == 4)
				return false;
			swapRow(M[i], M[j]);
		}
		for (j = i; j <= 3; j ++)
			divRow(M[j], M[j][i]);
		for (j = i + 1; j <= 3; j ++)
			if (!zero(M[j][i]))
				subtractRow(M[j], M[i]);
	}
	if (zero(M[3][3]))
		return false;

	w =  M[3][4] / M[3][3];
	z = (M[2][4] - M[2][3] * w) / M[2][2];
	y = (M[1][4] - M[1][3] * w - M[1][2] * z) / M[1][1];
	x = (M[0][4] - M[0][3] * w - M[0][2] * z - M[0][1] * y) / M[0][0];

	return true;
}

template<typename T>
__device__ __forceinline__
void MatrixSolver5<T>::swapRow(T r1[], T r2[]) {
	swapVal(r1[0], r2[0]);
	swapVal(r1[1], r2[1]);
	swapVal(r1[2], r2[2]);
	swapVal(r1[3], r2[3]);
	swapVal(r1[4], r2[4]);
}

template<typename T>
__device__ __forceinline__
void MatrixSolver5<T>::divRow(T r[], const T d) {
	if (zero(d))
		return;
	r[0] /= d;
	r[1] /= d;
	r[2] /= d;
	r[3] /= d;
	r[4] /= d;
}

template<typename T>
__device__ __forceinline__
void MatrixSolver5<T>::subtractRow(T r1[], const T r2[]) {
	r1[0] -= r2[0];
	r1[1] -= r2[1];
	r1[2] -= r2[2];
	r1[3] -= r2[3];
	r1[4] -= r2[4];
}

template<typename T>
__device__ __forceinline__
bool MatrixSolver5<T>::zero(const T &a) {
	return a < tolerance && a > -tolerance;
}

#endif //_ISO_MATH_CUDA_INC_