#ifndef HMESH_H
#define HMESH_H

#include <QtGui/QMainWindow>
#include "ui_hmesh.h"
#include "hGLWidget.h"
#include "dialog/qslimdialog.h"

class HMesh : public QMainWindow
{
	Q_OBJECT

public:
	HMesh(QWidget *parent = 0, Qt::WFlags flags = 0);
	~HMesh();

private:
	Ui::HMeshClass ui;

	hGlWidget *_hglwidget;
	QSlimDialog *_qslim_dialog;

	// variables for open files
	QString _prev_path;
	QString _file_name;
	QString _file_ext;

	// menus
	QMenu* _menu_file;			// file menu
		QAction* _action_open;
	QMenu* _menu_simp;			// simplification menu
		QAction* _action_qslim;
	QMenu* _menu_render;		// render menu
		QMenu* _menu_primitive;
			QAction* _action_wireframe;
			QAction* _action_flat_lines;
			QAction* _action_flat;
		QMenu* _menu_color_mode;
			QAction* _action_vert_color;
			QAction* _action_face_color;

public slots:
	void on_open_file();
	void on_qslim();

	void on_wireframe() { _hglwidget->primitiveMode(WIREFRAME); update(); }
	void on_flat_lines() { _hglwidget->primitiveMode(FLAT_LINES); update(); }
	void on_flat() { _hglwidget->primitiveMode(FLAT); }
	
	void on_vert_color() { _hglwidget->colorMode(VERT_COLOR); update(); }
	void on_face_color() { _hglwidget->colorMode(FACE_COLOR); update(); }
};

#endif // HMESH_H