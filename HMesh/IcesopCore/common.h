/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 定义公用数据及数据结构
** 基本宏定义
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

typedef BaseNode<double> Node;				// 节点类型
typedef BaseCollection<Node> Nodes; 		// 节点集合类型
typedef BaseElement Element;  				// 单元类型
typedef BaseCollection<Element> Elements; 	// 单元集合类型
typedef BaseMesh<double> Mesh;  			// 网格类型

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
