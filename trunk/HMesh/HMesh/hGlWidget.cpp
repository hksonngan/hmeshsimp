#include "hGlWidget.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include "common.h"
#include "icesop_common.h"
#include "ply/ply_inc.h"
#include "tri_soup_stream.h"
#include "h_math.h"
//#include "mesh_generator.h"
#include "glut.h"

using std::cout;
using std::cerr;
using std::endl;
using std::ostringstream;

#define _max_of_three(a, b, c, _max)  _max = max(a, b); _max = max(c, _max);

hGlWidget::hGlWidget()
{
	_drawWhich = NONE;

	_primitive_mode = FLAT_LINES;
	_color_mode = FACE_COLOR;
	_draw_tri_index = false;
	_light_on = false;

	//fnormals = NULL;
	//vnormals = NULL;

	//phverts = NULL;
	numvert = 0;
	//phfaces = NULL;
	numface = 0;

	initTransform();
	_max_x = 0;
	_min_x = 0;
	_max_y = 0;
	_min_y = 0;
	_max_z = 0;
	_min_z = 0;
	_center_x = 0;
	_center_y = 0;
	_center_z = 0;
}

hGlWidget::~hGlWidget() {
	//if (phverts)
	//	delete[] phverts;
	//if (phfaces)
	//	delete[] phfaces;
}

void hGlWidget::initializeGL()
{
	GLenum err = glewInit();
	if (GLEW_OK != err) 
		cerr << endl << "Error: " << glewGetErrorString(err) << endl;
	else 
		cout << endl << "Glew init success" << endl;

	cout << "Opengl Version: " << glGetString(GL_VERSION) << endl;

	//glShadeModel(GL_FLAT);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_NORMALIZE);

	// Default : blending
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);

	GLint buf, sbuf;
	glGetIntegerv(GL_SAMPLE_BUFFERS, &buf);
	cout << "Number of sample buffers: " << buf << endl;
	glGetIntegerv(GL_SAMPLES, &sbuf);
	cout << "Number of samples: " << sbuf << endl << endl;
	// antialiasing
	glEnable(GL_MULTISAMPLE);

	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClearDepth(1.0f);

	// specifies which buffer to draw into
	glDrawBuffer(GL_BACK);
}

void hGlWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// draw shaded triangles
	if (_primitive_mode != WIREFRAME) {
		glPolygonOffset(1.0, 1.0);
		glEnable(GL_POLYGON_OFFSET_FILL);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		//gluLookAt(0, 0, 2, 0, 0, 0, 0, 1, 0);

		if (_light_on)
			setLights();
		else
			setColor();

		applyTransform();

		setMaterial();
		//glPolygonMode(GL_FRONT, GL_FILL);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		drawModel();
		glDisable(GL_POLYGON_OFFSET_FILL);
	}

	// draw triangle lines
	if (_primitive_mode == FLAT_LINES || _primitive_mode == WIREFRAME) {
		glDisable(GL_LIGHTING);
		
		bool drawTriIndexIsOn = _draw_tri_index;
		_draw_tri_index = false;

		glColor3f(0.0f, 0.0f, 0.0f);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		//gluLookAt(0, 0, 2, 0, 0, 0, 0, 1, 0);

		applyTransform();
		//glPolygonMode(GL_FRONT, GL_LINE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		drawModel();

		_draw_tri_index = drawTriIndexIsOn;
	}
}

void hGlWidget::drawModel() {
	if(_drawWhich == DRAW_PLY){
		drawPly();
	} else if (_drawWhich == DRAW_TRIS) {
		drawTrisoup();
	} else if (_drawWhich == DRAW_MC_TRIS) {
		drawMCTrisoup();
	} else if (_drawWhich == DRAW_H_INDEXED_MESH || _drawWhich == DRAW_VERB_INDEXED_MESH) {
		drawIndexedMesh();
	}
}

void hGlWidget::resizeGL(int width, int height)
{
	glMatrixMode(GL_PROJECTION);
	glViewport(0, 0, width, height);
	//glPushMatrix();
	glLoadIdentity();
	gluPerspective(45.0f, (float)width / (float)height, 1.0f, 1000.0f);
}

void hGlWidget::initTransform() {
	memset (_glmat, 0, sizeof (_glmat));
	_glmat[0] = _glmat[5] = _glmat[10] = _glmat[15] = 1;
	_scale = 10.0;
	_operateMode = OPERATEMODE_NONE;
	_trans_point.setValue(0.0, 0.0, 0.0);
	double _rotate_degree = 0.0;
}

void hGlWidget::applyTransform() {
	glTranslatef(0.0, 0.0, -5.0);
	//glTranslatef(_trans_point.x, _trans_point.y, _trans_point.z);
	glScalef(_scale, _scale, _scale);
	//glTranslatef(0.0, 0.0, -_scale);
	// rotate
	glMultMatrixd(_glmat);
	// locate model to (0, 0, 0)
	glTranslatef(-_center_x, -_center_y, -_center_z);
}

void hGlWidget::setLights() {
	// Light 0 properties
	float ambientProperties[]	= {0.6f, 0.6f, 0.6f, 1.0f};
	float diffuseProperties[]	= {1.0f, 1.0f, 1.0f, 1.0f};
	float specularProperties[]	= {1.0f, 1.0f, 1.0f, 1.0f};
	float lightPosition[]		= {0.0f, 0.0f, 1.0f, 0.0f};

	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientProperties);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseProperties);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specularProperties);
	glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);

	// Light model
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);

	glEnable(GL_NORMALIZE);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
}

void hGlWidget::setColor() {
	glDisable(GL_LIGHTING);
	glColor3f(0.8, 0.8, 0.8);
}

void hGlWidget::setMaterial() {
	// material
	float	MatAmbient[]    = {0.4f, 0.4f, 0.4f, 1.0f};
	float	MatDiffuse[]    = {0.5f, 0.5f, 0.5f, 1.0f};
	//float	MatSpecular[]   = {0.0f, 0.0f, 0.0f, 1.0f};
	float	MatSpecular[]   = {0.17f, 0.17f, 0.17f, 1.0f};
	float	MatShininess[]  = { 60 };
	float	MatEmission[]   = {0.0f, 0.0f, 0.0f, 1.0f};

	glMaterialfv(GL_FRONT, GL_AMBIENT, MatAmbient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MatDiffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, MatSpecular);
	glMaterialfv(GL_FRONT, GL_SHININESS, MatShininess);
	glMaterialfv(GL_FRONT, GL_EMISSION, MatEmission);

	float	MatAmbientBack[]    = {0.05f, 0.05f, 0.05f, 1.0f};
	float	MatDiffuseBack[]    = {0.1f, 0.1f, 0.1f, 1.0f};
	float	MatSpecularBack[]   = {0.0f, 0.0f, 0.0f, 1.0f};
	float	MatEmissionBack[]   = {0.0f, 0.0f, 0.0f, 1.0f};

	glMaterialfv(GL_BACK, GL_AMBIENT, MatAmbientBack);
	glMaterialfv(GL_BACK, GL_DIFFUSE, MatDiffuseBack);
	glMaterialfv(GL_BACK, GL_SPECULAR, MatSpecularBack);
	glMaterialfv(GL_BACK, GL_EMISSION, MatEmission);
}

void hGlWidget::clearAllModels() {
    vnormals.clear();
    fnormals.clear();
    _tris_container.clear();
    _mc_tris.clear();
    vertVec.clear();
    faceVec.clear();
    numvert = 0;
    numface = 0;
    vertVecVerb.clear();
    faceVecVerb.clear();
    clean_ply();
}

void hGlWidget::setDrawQSlim()
{
	_drawWhich = DRAW_QSLIM;
	update();
}

void hGlWidget::setDrawPly()
{
	_drawWhich = DRAW_PLY;
	update();
}

void hGlWidget::setDrawTris()
{
	_drawWhich = DRAW_TRIS;
	update();
}

void hGlWidget::getMCTrisMaxMin() {
	_min_x = _max_x = _mc_tris[0];
	_min_y = _max_y = _mc_tris[1];
	_min_z = _max_z = _mc_tris[2];

	for (int i = 1; i < _mc_tris.size()/3; i ++) {
		if (_mc_tris[i*3] > _max_x)
			_max_x = _mc_tris[i*3];
		else if (_mc_tris[i*3] < _min_x)
			_min_x = _mc_tris[i*3];
		if (_mc_tris[i*3+1] > _max_y)
			_max_y = _mc_tris[i*3+1];
		else if (_mc_tris[i*3+1] < _min_y)
			_min_y = _mc_tris[i*3+1];
		if (_mc_tris[i*3+2] > _max_z)
			_max_z = _mc_tris[i*3+2];
		else if (_mc_tris[i*3+2] < _min_z)
			_min_z = _mc_tris[i*3+2];
	}
}

bool hGlWidget::setDrawMC(string filename, double isovalue, int* sampleStride) {
    VolumeSet* volSet;
    clearAllModels();

    ///////////////////////////////////////////////////////////////////////////////////
	MCSimp mcsimp;

	if (!mcsimp.genIsosurfaces(filename, isovalue, sampleStride, _mc_tris))
		return false;

	ofstream fout("gensimp.log", ofstream::app | ofstream::out);
	cout << mcsimp.info();
	fout << mcsimp.info();

	//volSet = mcsimp.getVolSet();
	//_max_x = volSet->thickness.s[0] * volSet->volumeSize.s[0]; _min_x = 0;
	//_max_y = volSet->thickness.s[1] * volSet->volumeSize.s[1]; _min_y = 0;
	//_max_z = volSet->thickness.s[2] * volSet->volumeSize.s[2]; _min_z = 0; 

	getMCTrisMaxMin();
	setBoundingBox();

    ///////////////////////////////////////////////////////////////////////////////////
	//VolumeSet _volSet;
	//if (!_volSet.parseDataFile(filename, true, false))
	//	return false;
	//MeshGenerator meshGen;
	//int sizes[3];
	//sizes[0] = _volSet.volumeSize.s[0];
	//sizes[1] = _volSet.volumeSize.s[1];
	//sizes[2] = _volSet.volumeSize.s[2];
	//float spacings[3];
	//spacings[0] = _volSet.thickness.s[0];
	//spacings[1] = _volSet.thickness.s[1];
	//spacings[2] = _volSet.thickness.s[2];
	//if (!meshGen.GenerateMesh(sizes, spacings, sampleStride, 
	//	reinterpret_cast<short *>(_volSet.getData()), isovalue, _mc_tris))
	//	return false;

	//volSet = &_volSet;
	//_max_x = volSet->thickness.s[0] * volSet->volumeSize.s[0]; _min_x = 0;
	//_max_y = volSet->thickness.s[1] * volSet->volumeSize.s[1]; _min_y = 0;
	//_max_z = 0; _min_z = -(volSet->thickness.s[2] * volSet->volumeSize.s[2]);
	//setBoundingBox();


    /////////////////////////////////////////////////////////////////////////////////////
	_drawWhich = DRAW_MC_TRIS;
	update();
	return true;
}

void hGlWidget::setDrawMCSimp(
    string filename, double isovalue, int* sampleStride, 
    double deimateRate, unsigned int maxNewTri) {
    VolumeSet* volSet;
    clearAllModels();

    /////////////////////////////////////////////////////////////////
	MCSimp mcsimp;
	if (!mcsimp.genCollapse(filename, isovalue, deimateRate, sampleStride, maxNewTri, numvert, numface)) {
		cerr << "#error occurred during simplification" << endl << endl;
		return;
	}
    vertVec.resize(numvert);
    faceVec.resize(numface);
    mcsimp.toIndexedMesh(vertVec.data(), faceVec.data());
	ofstream fout("gensimp.log", ofstream::app | ofstream::out);
	fout << mcsimp.info();
	cout << mcsimp.info();
    _drawWhich = DRAW_H_INDEXED_MESH;
    computeIndexMeshNormals();
    volSet = mcsimp.getVolSet();


    /////////////////////////////////////////////////////////////////
	//VolumeSet _volSet;
	//if (!_volSet.parseDataFile(filename, true, false))
	//	return;
	//MeshGenerator meshGen;
	//int sizes[3];
	//sizes[0] = _volSet.volumeSize.s[0];
	//sizes[1] = _volSet.volumeSize.s[1];
	//sizes[2] = _volSet.volumeSize.s[2];
	//float spacings[3];
	//spacings[0] = _volSet.thickness.s[0];
	//spacings[1] = _volSet.thickness.s[1];
	//spacings[2] = _volSet.thickness.s[2];
	//if (!meshGen.GenerateCollapse(sizes, spacings, sampleStride, 
	//	reinterpret_cast<short *>(_volSet.getData()), isovalue, 
	//	deimateRate, vertVecVerb, faceVecVerb, maxNewTri))
	//	return;
	//
	//numvert = vertVecVerb.size() / 3;
	//numface = faceVecVerb.size() / 3;
	//_drawWhich = DRAW_VERB_INDEXED_MESH;
	//computeIndexMeshNormals();
	//volSet = &_volSet;
	

    //////////////////////////////////////////////////////////////////
    // bounding boxes
	_max_x = volSet->thickness.s[0] * volSet->volumeSize.s[0]; _min_x = 0;
	_max_y = volSet->thickness.s[1] * volSet->volumeSize.s[1]; _min_y = 0;
	_max_z = volSet->thickness.s[2] * volSet->volumeSize.s[2]; _min_z = 0; 
	setBoundingBox();

	update();
}

void hGlWidget::mousePressEvent(QMouseEvent * event)
{
	QPoint currentButtonPos = event->pos();
	if(event->buttons() == Qt::LeftButton)
	{
		_lButtonPressPos = event->pos();

		switch(_operateMode)
		{
		case OPERATEMODE_NONE:
			_operateMode = OPERATEMODE_ROTATE;
			break;
		case OPERATEMODE_SELECT:
			break;
		case OPERATEMODE_PAN:
			break;
		case OPERATEMODE_ROTATE:
			break;
		case OPERATEMODE_ZOOM:
			break;
		default:
			break; 
		}
	}
	else if(event->buttons() == Qt::RightButton)
	{
		_rButtonPressPos = event->pos();

		switch(_operateMode)
		{
		case OPERATEMODE_NONE:
			_operateMode = OPERATEMODE_PAN;
			break;
		case OPERATEMODE_SELECT:
			break;
		case OPERATEMODE_PAN:
			break;
		case OPERATEMODE_ROTATE:
			break;
		case OPERATEMODE_ZOOM:
			break;
		default:
			break; 
		}
	}
}

void hGlWidget::mouseReleaseEvent(QMouseEvent * event)
{
	switch(_operateMode)
	{
	case OPERATEMODE_NONE:
		break;
	case OPERATEMODE_SELECT:
		break;
	case OPERATEMODE_PAN:
		_operateMode = OPERATEMODE_NONE;
		break;
	case OPERATEMODE_ROTATE:
		_operateMode = OPERATEMODE_NONE;
		break;
	case OPERATEMODE_ZOOM:
		break;
	default:
		break; 
	}
}

void hGlWidget::mouseMoveEvent(QMouseEvent * event)
{
	QPoint currentButtonPos = event->pos();

	switch(_operateMode)
	{
	case OPERATEMODE_NONE:
		break;
	case OPERATEMODE_SELECT:
		break;
	case OPERATEMODE_PAN:
		if(event->buttons() == Qt::RightButton)
		{
			GLint viewport[4];
			glGetIntegerv(GL_VIEWPORT, viewport);

			Vector3D vt((currentButtonPos.x() - _rButtonPressPos.x()), (- currentButtonPos.y() + _rButtonPressPos.y()), 0);
			_trans_point += vt * 0.005;

			_rButtonPressPos = currentButtonPos;
			update();
		}
		break;
	case OPERATEMODE_ROTATE:
		if(event->buttons() == Qt::LeftButton)
		{
			Matrix4 mat;
			glmat2mat(_glmat, mat);
			Matrix4 mat2 = ! mat;

			Vector3D vr(currentButtonPos.x() - _lButtonPressPos.x(), - currentButtonPos.y() + _lButtonPressPos.y(), 0);
			vr = Vector3D(0, 0, 1) ^ vr;
			vr = vr * mat2;

			double rotateSpeed = 1.0 / 3.0;
			double ang = (vr.Length() * rotateSpeed) * PI / 180;
			vr.SetUnit();
			mat2.SetRotateV(vr, ang);

			mat = mat2 * mat;
			mat2glmat(mat, _glmat);

			_lButtonPressPos = currentButtonPos;
			update();
		}
		break;
	case OPERATEMODE_ZOOM:
		if(event->buttons() == Qt::LeftButton)
		{
			double scaleSpeed = 1.0 / 300.0;
			double scale = 1.0 + (- currentButtonPos.y() + _lButtonPressPos.y()) * scaleSpeed;
			_scale *= scale;
			_lButtonPressPos = currentButtonPos;
			update();
		}
		break;
	default:
		break; 
	}

}

void hGlWidget::wheelEvent(QWheelEvent * event)
{
	//double scaleSpeed = _range / 500.0;
	//_scale += (event->delta() / 20) * scaleSpeed;

	double scaleSpeed = 1.0 / 300.0;
	double scale = 1.0 + (- event->delta() / 5) * scaleSpeed;
	_scale *= scale;

	update();
}

void hGlWidget::openFile(QString _file_name)
{
	QString _file_ext = _file_name.mid(_file_name.lastIndexOf(".") + 1);

	if (_file_ext.toLower() == "ply") {
        clearAllModels();
		ply_read_file(_file_name.toLocal8Bit().data());
		computePlyNormals();

		int vert_size = sizeof(Vertex) +  3 * sizeof(float); 
		int vertices_mem = vert_size * nverts;
		int face_size = sizeof(Face);
		int faces_mem = face_size * nfaces;
		cerr << endl << "\t== ply file statistics ==" << endl << "\tper vertex bytes : " << vert_size << endl;
		cerr << "\tvertices count : " << nverts << endl;
		cerr << "\tvertices bytes : " << vertices_mem << endl;
		cerr << "\tper face bytes : " << face_size << endl;
		cerr << "\tfaces count : " << nfaces << endl;
		cerr << "\tfaces bytes : " << faces_mem << endl;
		cerr << "\ttotal : " << faces_mem + vertices_mem << endl;
		setDrawPly();
		calcPlyBoundingBox();
	}
	else if (_file_ext.toLower() == "tris") {
		clearAllModels();
		if (_tris_container.read(_file_name.toLocal8Bit().data()) == false)
		{
			cerr << "#error: read triangle soup file failed" << endl;
			_tris_container.clear();
			return;
		}
		setDrawTris();
		//calcBoundingBoxTris();

		_max_x = _tris_container.getTriSStream()->getMaxX();
		_min_x = _tris_container.getTriSStream()->getMinX();
		_max_y = _tris_container.getTriSStream()->getMaxY();
		_min_y = _tris_container.getTriSStream()->getMinY();
		_max_z = _tris_container.getTriSStream()->getMaxZ();
		_min_z = _tris_container.getTriSStream()->getMinZ();

		setBoundingBox();
	} else if (_file_ext.toLower() == "_tris") {
		clearAllModels();
		std::ifstream fin(_file_name.toLocal8Bit().data());
		if (!fin.good()) {
			std::cout << "error open file" << std::endl;
		}

		float f;
		fin >> f;
		_max_x = _min_x = f;
		_mc_tris.push_back(f);
		fin >> f;
		_max_y = _min_y = f;
		_mc_tris.push_back(f);
		fin >> f;
		_max_z = _min_z = f;
		_mc_tris.push_back(f);

		while(true) {
			fin >> f;
			if (fin.eof())
				break;
			_mc_tris.push_back(f);
			if (f > _max_x)
				_max_x = f;
			else if (f < _min_x)
				_min_x = f;
			fin >> f;
			_mc_tris.push_back(f);
			if (f > _max_y)
				_max_y = f;
			else if (f < _min_y)
				_min_y = f;
			fin >> f;
			_mc_tris.push_back(f);
			if (f > _max_z)
				_max_z = f;
			else if (f < _min_z)
				_min_z = f;
		}

		_drawWhich = DRAW_MC_TRIS;

		//////////////////////////////////////
		// for debug
		//float x_center = 152;
		//float y_center = 129;
		//float z_center = 132;
		//float span = 6;
		//_min_x = x_center-span;
		//_max_x = x_center+span;
		//_min_y = y_center-span;
		//_max_y = y_center+span;
		//_min_z = z_center-span;
		//_max_z = z_center+span;
		//////////////////////////////////////

		setBoundingBox();
		update();
	} else if (_file_ext.toLower() == "rawtris") {
		clearAllModels();
		std::ifstream fin(_file_name.toLocal8Bit().data(), std::ios::in | std::ios::binary);
		if (!fin.good()) {
			std::cout << "error open file" << std::endl;
		}

		float f[3];
		fin.read(reinterpret_cast<char*>(f), 3*sizeof(float));
		_max_x = _min_x = f[0];
		_mc_tris.push_back(f[0]);
		_max_y = _min_y = f[1];
		_mc_tris.push_back(f[1]);
		_max_z = _min_z = f[2];
		_mc_tris.push_back(f[2]);

		int i = 1;
		while(true) {
			fin.read(reinterpret_cast<char*>(f), 3*sizeof(float));
			if (fin.eof())
				break;
			_mc_tris.push_back(f[0]);
			if (f[0] > _max_x)
				_max_x = f[0];
			else if (f[0] < _min_x)
				_min_x = f[0];
			_mc_tris.push_back(f[1]);
			if (f[1] > _max_y)
				_max_y = f[1];
			else if (f[1] < _min_y)
				_min_y = f[1];
			_mc_tris.push_back(f[2]);
			if (f[2] > _max_z)
				_max_z = f[2];
			else if (f[2] < _min_z)
				_min_z = f[2];

			i ++;
			//if (i >= 300)
				//break;
		}

		_drawWhich = DRAW_MC_TRIS;
		_draw_all_tri_index = false;

		//////////////////////////////////////
		// for debug
		//float x_center = 0.2;
		//float y_center = 4.2;
		//float z_center = 12.52;
		//float span = 2;
		//_min_x = x_center-span;
		//_max_x = x_center+span;
		//_min_y = y_center-span;
		//_max_y = y_center+span;
		//_min_z = z_center-span;
		//_max_z = z_center+span;
		//_draw_all_tri_index = true;
		//////////////////////////////////////

		setBoundingBox();
		update();
	}

	initTransform();
}

void hGlWidget::drawTrisoup() {
	int i;
	HVertex v1, v2, v3, vText;

	for (i = 0; i < _tris_container.count(); i ++) {
		if (_draw_tri_index && i % 30 == 0) {
			v1.Set(_tris_container(i).vert1.x, _tris_container(i).vert1.y, _tris_container(i).vert1.z);
			v2.Set(_tris_container(i).vert2.x, _tris_container(i).vert2.y, _tris_container(i).vert2.z);
			v3.Set(_tris_container(i).vert3.x, _tris_container(i).vert3.y, _tris_container(i).vert3.z);
			vText = getTriIndexTextPos(v1, v2, v3);
			showNum(vText.x, vText.y, vText.z, i);
		} 

		glBegin(GL_TRIANGLES);
		glVertex3f(_tris_container(i).vert1.x, _tris_container(i).vert1.y, _tris_container(i).vert1.z);
		glVertex3f(_tris_container(i).vert2.x, _tris_container(i).vert2.y, _tris_container(i).vert2.z);
		glVertex3f(_tris_container(i).vert3.x, _tris_container(i).vert3.y, _tris_container(i).vert3.z);
		glEnd();
	}
}

void hGlWidget::drawMCTrisoup() {
	HNormal nm;
	HVertex v1, v2, v3, vText, vMean;

	const float *tri_ptr = _mc_tris.data();
	float *data = _mc_tris.data();

	int i;
	for (i = 0; i < _mc_tris.size() / 9; i ++) {
		v1.Set(_mc_tris[i*9],   _mc_tris[i*9+1], _mc_tris[i*9+2]);
        v2.Set(_mc_tris[i*9+3], _mc_tris[i*9+4], _mc_tris[i*9+5]);
        v3.Set(_mc_tris[i*9+6], _mc_tris[i*9+7], _mc_tris[i*9+8]);
		nm = triangleNormal(v1, v2, v3);

		//if (i == 360 || i == 356)
		//	continue;

		vMean = (v1 + v2 + v3) / 3;
		if (vMean.x<_min_x || vMean.x>_max_x || vMean.y<_min_y || vMean.y>_max_y || 
			vMean.z<_min_z || vMean.z>_max_z) 
			continue;

		//if (_draw_tri_index && (_draw_all_tri_index || i % 30 == 0)) {
		if (_draw_tri_index && i % 20 == 1) {
			vText = getTriIndexTextPos(v1, v2, v3);
			showNum(vText.x, vText.y, vText.z, i);
		}

		glBegin(GL_TRIANGLES);
		//glNormal3f(nm.x, nm.y, nm.z);
		glNormal3f(-nm.x, -nm.y, -nm.z);
		glVertex3f(v1.x, v1.y, v1.z);
		glVertex3f(v2.x, v2.y, v2.z);
		glVertex3f(v3.x, v3.y, v3.z);

		//glNormal3f(-nm.x, -nm.y, -nm.z);
		//glVertex3f(v1.x, v1.y, v1.z);
		//glVertex3f(v2.x, v2.y, v2.z);
		//glVertex3f(v3.x, v3.y, v3.z);
		glEnd();
	}
}

void hGlWidget::drawPly() {
	Vertex v;
	Face f;

	glBegin(GL_TRIANGLES);

	for(int i = 0; i < nfaces; i ++)
	{
		f = flist[i];

		if( f.nverts == 3 ) {
			// draw front face
			if (_primitive_mode == FLAT || _primitive_mode == FLAT_LINES)
				glNormal3f(fnormals[i].x, fnormals[i].y, fnormals[i].z);

			if (_primitive_mode == SMOOTH) {
				HNormal &nm = vnormals[flist[i].verts[0]];
				glNormal3f(nm.x, nm.y, nm.z);
			}
			v = vlist[f.verts[0]];
			glVertex3f(v.x, v.y, v.z);

			if (_primitive_mode == SMOOTH) {
				HNormal &nm = vnormals[flist[i].verts[1]];
				glNormal3f(nm.x, nm.y, nm.z);
			}
			v = vlist[f.verts[1]];
			glVertex3f(v.x, v.y, v.z);

			if (_primitive_mode == SMOOTH) {
				HNormal &nm = vnormals[flist[i].verts[2]];
				glNormal3f(nm.x, nm.y, nm.z);
			}
			v = vlist[f.verts[2]];
			glVertex3f(v.x, v.y, v.z);

			// draw back face
			//if (_primitive_mode == FLAT || _primitive_mode == FLAT_LINES)
			//	glNormal3f(fnormals[i].x, fnormals[i].y, fnormals[i].z);

			//if (_primitive_mode == SMOOTH) {
			//	HNormal &nm = vnormals[flist[i].verts[0]];
			//	glNormal3f(nm.x, nm.y, nm.z);
			//}
			//v = vlist[f.verts[0]];
			//glVertex3f(v.x, v.y, v.z);

			//if (_primitive_mode == SMOOTH) {
			//	HNormal &nm = vnormals[flist[i].verts[2]];
			//	glNormal3f(nm.x, nm.y, nm.z);
			//}
			//v = vlist[f.verts[2]];
			//glVertex3f(v.x, v.y, v.z);

			//if (_primitive_mode == SMOOTH) {
			//	HNormal &nm = vnormals[flist[i].verts[1]];
			//	glNormal3f(nm.x, nm.y, nm.z);
			//}
			//v = vlist[f.verts[1]];
			//glVertex3f(v.x, v.y, v.z);
		}
		else {
			cerr << "#error! non-triangle while drawing ply models" << endl;
		}
	}

	glEnd();
}

void hGlWidget::drawIndexedMesh() {
	HVertex v1, v2, v3;
	int j;
    HFace f;

	glBegin(GL_TRIANGLES);
	for(int i = 0; i < numface; i ++) {
        if (_drawWhich == DRAW_H_INDEXED_MESH)
		    f = faceVec[i];

		// draw front face
		if (_primitive_mode == FLAT || _primitive_mode == FLAT_LINES)
			glNormal3f(fnormals[i].x, fnormals[i].y, fnormals[i].z);

		switch (_drawWhich) {
		case DRAW_H_INDEXED_MESH:
			j = f.i;
			v1 = vertVec[f.i];
			break;
		case DRAW_VERB_INDEXED_MESH:
			j = faceVecVerb[i * 3];
			v1.Set(vertVecVerb[j * 3], vertVecVerb[j * 3 + 1], vertVecVerb[j * 3 + 2]);
			break;
		}
		if (_primitive_mode == SMOOTH) {
			HNormal &nm = vnormals[j];
			glNormal3f(nm.x, nm.y, nm.z);
		}
		glVertex3f(v1.x, v1.y, v1.z);

		switch (_drawWhich) {
		case DRAW_H_INDEXED_MESH:
			j = f.j;
			v2 = vertVec[f.j];
			break;
		case DRAW_VERB_INDEXED_MESH:
			j = faceVecVerb[i * 3 + 1];
			v2.Set(vertVecVerb[j * 3], vertVecVerb[j * 3 + 1], vertVecVerb[j * 3 + 2]);
			break;
		}
		if (_primitive_mode == SMOOTH) {
			HNormal &nm = vnormals[j];
			glNormal3f(nm.x, nm.y, nm.z);
		}
		glVertex3f(v2.x, v2.y, v2.z);

		switch (_drawWhich) {
		case DRAW_H_INDEXED_MESH:
			j = f.k;
			v3 = vertVec[f.k];
			break;
		case DRAW_VERB_INDEXED_MESH:
			j = faceVecVerb[i * 3 + 2];
			v3.Set(vertVecVerb[j * 3], vertVecVerb[j * 3 + 1], vertVecVerb[j * 3 + 2]);
			break;
		}
		if (_primitive_mode == SMOOTH) {
			HNormal &nm = vnormals[j];
			glNormal3f(nm.x, nm.y, nm.z);
		}
		glVertex3f(v3.x, v3.y, v3.z);

		// draw front face
		//if (_primitive_mode == FLAT || _primitive_mode == FLAT_LINES)
		//	glNormal3f(fnormals[i].x, fnormals[i].y, fnormals[i].z);

		//if (_primitive_mode == SMOOTH) {
		//	HNormal &nm = vnormals[f.i];
		//	glNormal3f(nm.x, nm.y, nm.z);
		//}
		//HVertex &v4 = vertVec[f.i];
		//glVertex3f(v4.x, v4.y, v4.z);

		//if (_primitive_mode == SMOOTH) {
		//	HNormal &nm = vnormals[f.k];
		//	glNormal3f(nm.x, nm.y, nm.z);
		//}
		//HVertex &v5 = vertVec[f.k];
		//glVertex3f(v5.x, v5.y, v5.z);

		//if (_primitive_mode == SMOOTH) {
		//	HNormal &nm = vnormals[f.j];
		//	glNormal3f(nm.x, nm.y, nm.z);
		//}
		//HVertex &v6 = vertVec[f.j];
		//glVertex3f(v6.x, v6.y, v6.z);
	}

	glEnd();
}

void hGlWidget::computePlyNormals() {
	int i = 0;
	HVertex v1, v2, v3;
	HNormal e1, e2, nm;

	fnormals.resize(nfaces);
	vnormals.resize(nverts);
	for (i = 0; i < nverts; i ++)
		vnormals[i].Set(0, 0, 0);

	for (i = 0; i < nfaces; i ++) {
		v1.Set(vlist[flist[i].verts[0]].x, vlist[flist[i].verts[0]].y, vlist[flist[i].verts[0]].z);
		v2.Set(vlist[flist[i].verts[1]].x, vlist[flist[i].verts[1]].y, vlist[flist[i].verts[1]].z);
		v3.Set(vlist[flist[i].verts[2]].x, vlist[flist[i].verts[2]].y, vlist[flist[i].verts[2]].z);

		e1 = v3 - v1;
		e2 = v2 - v1;
		nm = e1 ^ e2;

		nm = -nm;

		vnormals[flist[i].verts[0]] += nm;
		vnormals[flist[i].verts[1]] += nm;
		vnormals[flist[i].verts[2]] += nm;

		nm.Normalize();
		fnormals[i] = nm;
	}

	for (i = 0; i < nverts; i ++)
		vnormals[i].Normalize();
}

void hGlWidget::computeIndexMeshNormals() {
	int i = 0, vi1, vi2, vi3;
	HNormal e1, e2, nm;
	HVertex v1, v2, v3;

	fnormals.resize(numface);
	vnormals.resize(numvert);
	for (i = 0; i < numvert; i ++)
		vnormals[i].Set(0, 0, 0);

	for (i = 0; i < numface; i ++) {
		switch (_drawWhich) {
		case DRAW_H_INDEXED_MESH:
            vi1 = faceVec[i].i;
            vi2 = faceVec[i].j;
            vi3 = faceVec[i].k;
			v1 = vertVec[vi1];
			v2 = vertVec[vi2];
			v3 = vertVec[vi3];
			break;
		case DRAW_VERB_INDEXED_MESH:
			vi1 = faceVecVerb[i * 3];
			v1.Set(vertVecVerb[vi1 * 3], vertVecVerb[vi1 * 3 + 1], vertVecVerb[vi1 * 3 + 2]);
			vi2 = faceVecVerb[i * 3 + 1];
			v2.Set(vertVecVerb[vi2 * 3], vertVecVerb[vi2 * 3 + 1], vertVecVerb[vi2 * 3 + 2]);
			vi3 = faceVecVerb[i * 3 + 2];
			v3.Set(vertVecVerb[vi3 * 3], vertVecVerb[vi3 * 3 + 1], vertVecVerb[vi3 * 3 + 2]);
			break;
		}

		e1 = v3 - v1;
		e2 = v2 - v1;
		nm = e1 ^ e2;

		nm = -nm;

		vnormals[vi1] += nm;
		vnormals[vi2] += nm;
		vnormals[vi3] += nm;

		nm.Normalize();
		fnormals[i] = nm;
	}

	for (i = 0; i < numvert; i ++)
		vnormals[i].Normalize();	
}

void hGlWidget::setBoundingBox() {
	_center_x = (_max_x + _min_x) / 2;
	_center_y = (_max_y + _min_y) / 2;
	_center_z = (_max_z + _min_z) / 2;
	_max_of_three(_max_x - _min_x, _max_y - _min_y, _max_z - _min_z, _range);

	//_scale = _range * 5;
	//_scale = 1;
	_scale = 0.2 / _range;

	cout << "bound box:" << endl
		<< "x: " << _min_x << " " << _max_x << endl
		<< "y: " << _min_y << " " << _max_y << endl
		<< "z: " << _min_z << " " << _max_z << endl << endl;
}

void hGlWidget::calcPlyBoundingBox() {
	if (nverts <= 0)
		return;

	_max_x = vlist[0].x;
	_min_x = vlist[0].x;
	_max_y = vlist[0].y;
	_min_y = vlist[0].y;
	_max_z = vlist[0].z;
	_min_z = vlist[0].z;

	for(int i = 1; i < nverts; i ++)
	{
		if (vlist[i].x > _max_x) {
			_max_x = vlist[i].x;
		}
		if (vlist[i].x < _min_x) {
			_min_x = vlist[i].x;
		}

		if (vlist[i].y > _max_y) {
			_max_y = vlist[i].y;
		}
		if (vlist[i].y < _min_y) {
			_min_y = vlist[i].y;
		}

		if (vlist[i].z > _max_z) {
			_max_z = vlist[i].z;
		}
		if (vlist[i].z < _min_z) {
			_min_z = vlist[i].z;
		}
	}

	setBoundingBox();
}

void hGlWidget::showText(float x, float y, float z, char *text) {
	bool LIGHT_IS_ON = false;
	if (glIsEnabled(GL_LIGHTING) == GL_TRUE) {
		LIGHT_IS_ON = true;
		glDisable(GL_LIGHTING);
	}

	glColor3f(0.8, 0.2, 0.2);
	glRasterPos3f(x, y, z);
	for (int i = 0; text[i]; i ++) {
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, text[i]);
	}

	if (LIGHT_IS_ON)
		glEnable(GL_LIGHTING);
	else 
		glColor3f(0.8, 0.8, 0.8);
}

void hGlWidget::showNum(float x, float y, float z, int n) {
	char str[50];
	sprintf(str, "%d", n);
	showText(x, y, z, str);
}

void hGlWidget::setDrawTriIndex() {
	if (_draw_tri_index == false) {
		_draw_tri_index = true;
	} else {
		_draw_tri_index = false;
		glDisable(GL_LIGHTING);
	}
}

void hGlWidget::setLightOnOff() {
	if (_light_on == true)
		_light_on = false;
	else {
		_light_on = true;
		glEnable(GL_LIGHTING);
	}
}

HVertex hGlWidget::getTriIndexTextPos(HVertex v1, HVertex v2, HVertex v3) {
	HVertex pos = (v1+v2+v3)/3;
	HVertex e1, e2, nm;
	e1 = v3 - v1;
	e2 = v2 - v1;
	nm = e1 ^ e2;
	nm.Normalize();
	pos += _range / 40 * nm;

	return pos;
}