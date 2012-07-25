/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ���������л�����ͨ�ñ���������
**
** Author : shan @2011
**
****************************************************************************/

#ifndef MESH_COMMON_H
#define MESH_COMMON_H

namespace icesop {

	
// Status_Null:ֻ����edge�����������ת�Ĺ�����
// Status_WireFrame:��������edge��������
// Status_HideLineInGray:��������edge�������ߣ����б��ڵ���edge�û�ɫ
// Status_HideLineRemoved:���ƴ�������edge��������
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
