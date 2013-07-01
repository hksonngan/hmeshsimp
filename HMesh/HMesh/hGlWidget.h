/*
 *  Qt OpenGL Widget For Displaying Meshes
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef H_GL_WIDGET
#define H_GL_WIDGET

#include <GL/glew.h>

#include <string>
#include <vector>
#include <QtGui>
#include <QGLWidget>
#include <QString>
#include "common.h"
#include "tri_soup.h"
#include "common_types.h"
#include "mc_simp.h"
#include "mc.h"

using namespace icesop;

using std::vector;
using std::string;

enum PrimitiveMode { SMOOTH, FLAT, FLAT_LINES, WIREFRAME };
enum ColorMode { VERT_COLOR, FACE_COLOR };
enum DrawWhich { 
	NONE, DRAW_PLY, DRAW_TRIS, DRAW_QSLIM, DRAW_MC_TRIS, 
	DRAW_H_INDEXED_MESH, DRAW_VERB_INDEXED_MESH };

// Qt OpenGL Widget For Displaying Meshes
class hGlWidget : public QGLWidget {
	Q_OBJECT

protected:
	void initializeGL();
	void paintGL();
	void resizeGL(int width, int height);
	void setGlState();

	void mousePressEvent(QMouseEvent * event);
	void mouseReleaseEvent(QMouseEvent * event);
	void mouseMoveEvent(QMouseEvent * event);
	void wheelEvent(QWheelEvent * event);

private:
	void initTransform();
	void computePlyNormals();
	void computeIndexMeshNormals();
	void setBoundingBox();
	void calcPlyBoundingBox();
	void drawModel();
	void drawTrisoup();
	void drawMCTrisoup();
	void drawPly();
	void drawIndexedMesh();
	void setLights();
	void setColor();
	void setMaterial();
	void applyTransform();
    void clearAllModels();
	void showText(float x, float y, float z, char *text);
	void showNum(float x, float y, float z, int n);
	void getMCTrisMaxMin();
	HVertex getTriIndexTextPos(HVertex v1, HVertex v2, HVertex v3);

public slots:
	void setDrawMCSimp(string filename, double isovalue, 
        int* sampleStride, double deimateRate, unsigned int maxNewTri);
	bool setDrawMC(string filename, double isovalue, int* sampleStride);

public:
	hGlWidget();
	~hGlWidget();
	void setDrawQSlim();
	void setDrawPly();
	void setDrawTris();
	void openFile(QString _file_name);

	void primitiveMode(PrimitiveMode m) { _primitive_mode = m; }
	void colorMode(ColorMode m) { _color_mode = m; }
	void setDrawTriIndex();
	void setLightOnOff();

private:
	// variables concerning drawing objects
	DrawWhich _drawWhich;

	PrimitiveMode	_primitive_mode;
	ColorMode		_color_mode;
	bool			_draw_tri_index;
	bool			_draw_all_tri_index;
	bool			_light_on;

	// info about bounding box
	float _max_x, _min_x;
	float _max_y, _min_y;
	float _max_z, _min_z;
	float _center_x, _center_y, _center_z;
	float _range;

	double		_glmat[16];		// translation matrix
	QPoint		_lButtonPressPos;
	QPoint		_rButtonPressPos;
	double		_scale;
	int			_operateMode;
	Vector3D	_trans_point;
	double		_rotate_degree;

	vector<HNormal>	vnormals;
	vector<HNormal>	fnormals;

	TriangleSoupContainer	_tris_container;
	vector<float>           _mc_tris;
	vector<HVertex>			vertVec;
	vector<HFace>			faceVec;
	unsigned int			numvert;
	unsigned int			numface;
	vector<float>			vertVecVerb;
	vector<int>				faceVecVerb;
};

#endif