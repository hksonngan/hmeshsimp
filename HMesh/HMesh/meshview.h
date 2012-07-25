/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 类MeshView:
** 定义一个窗口,
** 用于显示网格,并对其进行简单操作
**
** Author : shan @2011
**
****************************************************************************/

#ifndef MeshView_H
#define MeshView_H

#include "../mdichild.h"
#include "../../../mesh/meshdrawer.h"

#include <QtGui>
#include <QGLWidget>

namespace icesop {

class MeshView : public QGLWidget, public MdiChild
{
	Q_OBJECT
public:
	MeshView(QWidget* parent = 0, const QString& title = tr("网格模型"));
	MeshView(const QString& filename, QWidget* parent = 0, const QString& title = tr("网格模型"));
	~MeshView();

	int getMdiType();

public slots:
	void onSetOperateMode(int mode);
	void onSetZoomMode(int mode);

	bool onOpen();
	bool onSave();
	bool onSaveAs();

protected:
	void initializeGL();
	void paintGL();
	void resizeGL(int width, int height);

	void mouseDoubleClickEvent(QMouseEvent * event);
	void mousePressEvent(QMouseEvent * event);
	void mouseReleaseEvent(QMouseEvent * event);
	void mouseMoveEvent(QMouseEvent * event);
	void wheelEvent(QWheelEvent * event);
	void keyPressEvent(QKeyEvent * event);
	void keyReleaseEvent(QKeyEvent * event);

private:
	// 绘制背景
	void paintBkgnd();

	// 绘制坐标系
	void paintCoord();

	// 绘制字符串
	void print_bitmap_string(void* font, const char* s);
	void glPrint(const char *fmt, ...);

	void SetScale(double scale);

	int operateMode_;		// 操作模式：旋转、缩放、平移等
	int zoomMode_;
	QPoint lButtonPressPos_, rButtonPressPos_;
	QPoint currentButtonPos_;


	Mesh* mesh_;			// 网格信息
	MeshDrawer meshDrawer_;

	double zNear_, zFar_;	// 模型Z方向上显示时的距离
	double zInit_;
	double range_;
	double scale_;
	double aspect_;

	double glmat_[16];		// 转换矩阵

	/*
	 * todo: 现在阶段并未使用
	 *       添加这些属性
	 *       以后方便扩展
	 */
	MeshStatus viewstatus_;
	MeshVMode viewmode_;
};

} // namespace icesop

#endif // MeshView_H
