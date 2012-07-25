/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ��MeshView:
** ����һ������,
** ������ʾ����,��������м򵥲���
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
	MeshView(QWidget* parent = 0, const QString& title = tr("����ģ��"));
	MeshView(const QString& filename, QWidget* parent = 0, const QString& title = tr("����ģ��"));
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
	// ���Ʊ���
	void paintBkgnd();

	// ��������ϵ
	void paintCoord();

	// �����ַ���
	void print_bitmap_string(void* font, const char* s);
	void glPrint(const char *fmt, ...);

	void SetScale(double scale);

	int operateMode_;		// ����ģʽ����ת�����š�ƽ�Ƶ�
	int zoomMode_;
	QPoint lButtonPressPos_, rButtonPressPos_;
	QPoint currentButtonPos_;


	Mesh* mesh_;			// ������Ϣ
	MeshDrawer meshDrawer_;

	double zNear_, zFar_;	// ģ��Z��������ʾʱ�ľ���
	double zInit_;
	double range_;
	double scale_;
	double aspect_;

	double glmat_[16];		// ת������

	/*
	 * todo: ���ڽ׶β�δʹ��
	 *       �����Щ����
	 *       �Ժ󷽱���չ
	 */
	MeshStatus viewstatus_;
	MeshVMode viewmode_;
};

} // namespace icesop

#endif // MeshView_H
