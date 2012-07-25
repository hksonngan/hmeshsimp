/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ���幫�����ݼ����ݽṹ
** �����궨��
**
** Author : shan @2011
**
****************************************************************************/

#ifndef COMMON_H
#define COMMON_H

#include "datastructures/point.h"
#include "datastructures/bounds.h"
#include "datastructures/matrix.h"

#include "datastructures/basecollection.h"
#include "datastructures/basenode.h"
#include "datastructures/baseelement.h"
#include "datastructures/basemesh.h"

#include "datastructures/geometryset.h"

#include "math/mathlib.h"

namespace icesop {

typedef BaseNode<double> Node;				// �ڵ�����
typedef BaseCollection<Node> Nodes; 		// �ڵ㼯������
typedef BaseElement Element;  				// ��Ԫ����
typedef BaseCollection<Element> Elements; 	// ��Ԫ��������
typedef BaseMesh<double> Mesh;  			// ��������

#ifdef WIN32
#define ICESOP_EXPORT __declspec(dllexport)
#else
#define ICESOP_EXPORT
#endif

#ifdef WIN32
#define ICESOP_IMPORT __declspec(dllimport)
#else
#define ICESOP_IMPORT
#endif


} // namespace icesop

#endif // COMMON_H
