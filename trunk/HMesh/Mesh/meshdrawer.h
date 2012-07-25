/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 类MeshDrawer:
** 网格绘制器
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

	// 设置选择集合
	void setCurrentSelectedSet(GeometrySet* set);
	// 获取选择集合
	GeometrySet* getCurrentSelectedSet();


	// 设置被选中时的颜色 0.0 < r,g,b < 1.0
	void setPickedColor(float r, float g, float b);

	// 设置线条颜色  0.0 < r,g,b < 1.0
	void setColor(float r, float g, float b, float alpha = 1.0);

	// 设置线条宽度
	void setLineWidth(float w);

	// 设置剖视面参数
	void setPlane(double a, double b, double c, double d);

	// 设置及获取包围盒
	void setBounds(const Bounds& bounds);
	void setBounds(const Point3D& pmin, const Point3D& pmax);
	Bounds getBounds();

	// 绘制体网格
	void drawMesh(const Mesh* mesh, const MeshStatus status = Status_Null, GeometrySet* currentSelectedSet = NULL);

private:
	//判断某点,线面是否在当前被选中的集合中
	bool isFaceSelected(int face, GeometrySet* selectedSet) const;
	bool isVertexSelected(int vertex, GeometrySet* selectedSet) const;
	bool isEdgeSelected(int edge, GeometrySet* selectedSet) const;

	// 获得mesh中与平面 ax + by +cz = d 相交的element编号，存入 vector<Gluint> hitElements中
	// 用OpenGL的Picking实现，优点是平面的参数任意，将来实现旋转时可用。目前还有错...待调试
	void getHitElements(const Mesh* mesh, std::vector<GLuint>& hitElements, double a, double b, double c, double d);

	// 用正常的遍历实现，目前仅支持法向量为x,y,z方向的平面
	void getCutElements(const Mesh* mesh, std::vector<GLuint>& cutElements, double a, double b, double c, double d);

	// 获取体网格剖切时应绘制的表面三角网格
	void getShowSurfaceElements(const Mesh* mesh, std::vector<GLuint>& surfaceElements, double a, double b, double c, double d);

	// 绘制剖视面,注意一定要在画完网格之后调用
	void drawCuttingPlane();

	// 绘制节点
	void drawNodes(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

	// 绘制三角网格
	void drawTriangleSurface(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

	// 绘制四面体
	void drawTetra(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

	// 绘制四面体网格的线框模型
	void drawTetraWire(const Mesh* mesh, GeometrySet * currentSelectedSet = NULL);

	// 绘制四面体网格的消隐模型
	void drawTetraHides(const Mesh* mesh, std::vector<GLuint>& hitElements, std::vector<GLuint>& hitSurfElements,
						double a = 1, double b = 0, double c = 0, double d = 0);

	// 绘制六面体
	void drawHex(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

	// 绘制六面体网格的线框模型
	void drawHexWire(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

	// 绘制六面体网格的消隐模型
	void drawHexHides(const Mesh* mesh, std::vector<GLuint>& hitElements, std::vector<GLuint>& hitSurfElements,
					  double a = 1, double b = 0, double c = 0, double d = 0);


	// 绘制三棱柱
	void drawTriPrism(const Mesh* mesh, GeometrySet* currentSelectedSet = NULL);

private:
	Mesh* mesh_;		// 网格信息

	float colorR_, colorG_, colorB_, lineAlpha_;// 网格线颜色
	float pickedR_, pickedG_, pickedB_;			// 选中状态颜色设置
	float lineWidth_;	// 网格线宽度

	double cuttingPlaneA_, cuttingPlaneB_, cuttingPlaneC_, cuttingPlaneD_;		// 剖视面 Ax + By + Cz = D

	Bounds bounds_;		// 轴对齐包围盒

	GeometrySet* currentSelectedSet_;	// 当前选择的点、线、面集合
};

} // namespace icesop

#endif // MESHDRAWER_H
