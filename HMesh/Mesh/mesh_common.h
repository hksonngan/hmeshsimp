/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 定义网格中基本的通用变量、操作
**
** Author : shan @2011
**
****************************************************************************/

#ifndef MESH_COMMON_H
#define MESH_COMMON_H

namespace icesop {

	
// Status_Null:只绘制edge，用于鼠标旋转的过程中
// Status_WireFrame:绘制所有edge和轮廓线
// Status_HideLineInGray:绘制所有edge和轮廓线，其中被遮挡的edge用灰色
// Status_HideLineRemoved:绘制带消隐的edge和轮廓线
enum MeshStatus {
	Status_Null,
	Status_NodesOnly,
	Status_WireFrame,
	Status_HideLineInGray,
	Status_HideLineRemoved,
	Status_ShadedEdge,
	Status_Shaded
};
enum MeshVMode {
	Mesh_vNormal,
	Mesh_vSection,
	Mesh_vPerspective
};

} // namespace icesop

#endif // MESH_COMMON_H
