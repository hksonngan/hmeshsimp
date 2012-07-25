/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ��MeshDrawer:
** ���������
**
** Author : shan @2011
**
****************************************************************************/

#ifndef MESHDRAWER_H
#define MESHDRAWER_H

#include "mesh_global.h"
#include "mesh_common.h"
#include "../IcesopCore/common.h"

#include <QtOpenGL>

namespace icesop {

class MESH_EXPORT MeshDrawer
{
public:
	MeshDrawer();
	MeshDrawer(Mesh* mesh);
	MeshDrawer(const std::string& filename);
	~MeshDrawer();

	void setMesh(Mesh* mesh);
	void setMesh(const std::string& filename);
	Mesh* getMesh();

	// ����ѡ�񼯺�
	void setCurrentSelectedSet(GeometrySet* set);
	// ��ȡѡ�񼯺�
	GeometrySet* getCurrentSelectedSet();


	// ���ñ�ѡ��ʱ����ɫ 0.0 < r,g,b < 1.0
	void setPickedColor(float r, float g, float b);

	// ����������ɫ  0.0 < r,g,b < 1.0
	void setColor(float r, float g, float b, float alpha = 1.0);

	// �����������
	void setLineWidth(float w);

	// �������������
	void setPlane(double a, double b, double c, double d);

	// ���ü���ȡ��Χ��
	void setBounds(const Bounds& bounds);
	void setBounds(const Point3D& pmin, const Point3D& pmax);
	Bounds getBounds();

	// ����������
	void drawMesh(const Mesh* mesh, const MeshStatus status = Status_Null, GeometrySet* currentSelectedSet = NULL);

private:
	//�ж�ĳ��,�����Ƿ��ڵ�ǰ��ѡ�еļ�����
	bool isFaceSelected(int face, GeometrySet* selectedSet) const;
	bool isVertexSelected(int vertex, GeometrySet* selectedSet) const;
	bool isEdgeSelected(int edge, GeometrySet* selectedSet) const;

	// ���mesh����ƽ�� ax + by +cz = d �ཻ��element��ţ����� vector<Gluint> hitElements��
	// ��OpenGL��Pickingʵ�֣��ŵ���ƽ��Ĳ������⣬����ʵ����תʱ���á�Ŀǰ���д�...������
	void getHitElements(const Mesh* mesh, std::vector<GLuint>& hitElements, double a, double b, double c, double d);

	// �������ı���ʵ�֣�Ŀǰ��֧�ַ�����Ϊx,y,z�����ƽ��
	void getCutElements(const Mesh* mesh, std::vector<GLuint>& cutElements, double a, double b, double c, double d);

	// ��ȡ����������ʱӦ���Ƶı�����������
	void getShowSurfaceElements(const Mesh* mesh, std::vector<GLuint>& surfaceElements, double a, double b, double c, double d);

	// ����������,ע��һ��Ҫ�ڻ�������֮�����
	void drawCuttingPlane();

	// ���ƽڵ�
	void drawNodes(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

	// ������������
	void drawTriangleSurface(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

	// ����������
	void drawTetra(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

	// ����������������߿�ģ��
	void drawTetraWire(const Mesh* mesh, GeometrySet * currentSelectedSet = NULL);

	// �������������������ģ��
	void drawTetraHides(const Mesh* mesh, std::vector<GLuint>& hitElements, std::vector<GLuint>& hitSurfElements,
						double a = 1, double b = 0, double c = 0, double d = 0);

	// ����������
	void drawHex(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

	// ����������������߿�ģ��
	void drawHexWire(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

	// �������������������ģ��
	void drawHexHides(const Mesh* mesh, std::vector<GLuint>& hitElements, std::vector<GLuint>& hitSurfElements,
					  double a = 1, double b = 0, double c = 0, double d = 0);


	// ����������
	void drawTriPrism(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

private:
	Mesh* mesh_;		// ������Ϣ

	float colorR_, colorG_, colorB_, lineAlpha_;// ��������ɫ
	float pickedR_, pickedG_, pickedB_;			// ѡ��״̬��ɫ����
	float lineWidth_;	// �����߿��

	double cuttingPlaneA_, cuttingPlaneB_, cuttingPlaneC_, cuttingPlaneD_;		// ������ Ax + By + Cz = D

	Bounds bounds_;		// ������Χ��

	GeometrySet* currentSelectedSet_;	// ��ǰѡ��ĵ㡢�ߡ��漯��
};

} // namespace icesop

#endif // MESHDRAWER_H
