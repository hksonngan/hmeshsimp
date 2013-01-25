#include "hGlWidget.h"

#include <algorithm>
#include "common.h"
#include "icesop_common.h"
#include "ply/ply_inc.h"
#include "tri_soup_stream.h"
#include "h_math.h"

using std::cout;
using std::cerr;
using std::endl;

#define _max_of_three(a, b, c, _max)  _max = max(a, b); _max = max(c, _max);

hGlWidget::hGlWidget()
{
	_drawWhich = NONE;

	_primitive_mode = FLAT_LINES;
	_color_mode = FACE_COLOR;

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
	glEnable(GL_CULL_FACE);

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClearDepth(1.0f);

	// specifies which buffer to draw into
	glDrawBuffer(GL_BACK);
}

void hGlWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (_primitive_mode != WIREFRAME) {
		glPolygonOffset(1.0, 1.0);
		glEnable(GL_POLYGON_OFFSET_FILL);
		setLights();
		applyTransform();
		setMaterial();
		glPolygonMode(GL_FRONT, GL_FILL);
		drawModel();
		glDisable(GL_POLYGON_OFFSET_FILL);
	}

	if (_primitive_mode == FLAT_LINES || _primitive_mode == WIREFRAME) {
		glDisable(GL_LIGHTING);
		glColor3f(0.0f, 0.0f, 0.0f);
		applyTransform();
		glPolygonMode(GL_FRONT, GL_LINE);
		drawModel();
	}
}

void hGlWidget::drawModel() {
	if(_drawWhich == DRAW_PLY){
		drawPly();
	} else if (_drawWhich == DRAW_TRIS) {
		drawTrisoup();
	} else if (_drawWhich == DRAW_MC_TRIS) {
		drawMCTrisoup();
	} else if (_drawWhich == DRAW_H_INDEXED_MESH) {
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

bool hGlWidget::setDrawMC(std::string filename, double isovalue) {
	_mc_tris.clear();
	MCSimp mcsimp;

	if (!mcsimp.genIsosurfaces(filename, isovalue, _mc_tris))
		return false;

	std::cout << "#iso surfaces gened, faces count: " << _mc_tris.size() << std::endl;

	// bounding boxes
	VolumeSet* volSet = mcsimp.getVolSet();
	_max_x = volSet->thickness.s[0] * volSet->volumeSize.s[0]; _min_x = 0;
	_max_y = volSet->thickness.s[1] * volSet->volumeSize.s[1]; _min_y = 0;
	_max_z = 0; _min_z = -(volSet->thickness.s[2] * volSet->volumeSize.s[2]);
	setBoundingBox();

	_drawWhich = DRAW_MC_TRIS;
	update();
	return true;
}

bool hGlWidget::setDrawMCSimp(std::string filename, double isovalue, double deimateRate) {
	MCSimp mcsimp;

	if (!mcsimp.genCollapse(filename, isovalue, 0.25, 2000, numvert, numface)) {
		cerr << "#error occurred during simplification" << endl << endl;
		return false;
	}

	vertVec.resize(numvert);
	faceVec.resize(numface);
	mcsimp.toIndexedMesh(vertVec.data(), faceVec.data());
	computeIndexMeshNormals();

	cout << "#iso-surfaces decimated" << endl
		<< "#generated faces: " << mcsimp.getGenFaceCount() << ", vertices: " << mcsimp.getGenVertCount() << endl
		<< "#simplified faces: " << numface << ", vertices: " << numvert << endl << endl;

	// bounding boxes
	VolumeSet* volSet = mcsimp.getVolSet();
	_max_x = volSet->thickness.s[0] * volSet->volumeSize.s[0]; _min_x = 0;
	_max_y = volSet->thickness.s[1] * volSet->volumeSize.s[1]; _min_y = 0;
	_max_z = 0; _min_z = -(volSet->thickness.s[2] * volSet->volumeSize.s[2]);
	setBoundingBox();

	_drawWhich = DRAW_H_INDEXED_MESH;
	update();
	return true;
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
	double scaleSpeed = 1.0 / 300.0;
	double scale = 1.0 + (- event->delta() / 5) * scaleSpeed;
	_scale *= scale;

	update();
}

void hGlWidget::openFile(QString _file_name)
{
	QString _file_ext = _file_name.mid(_file_name.lastIndexOf(".") + 1);

	if (_file_ext.toLower() == "ply") {
		clean_ply();
		_tris_container.clear();
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
		clean_ply();
		_tris_container.init();
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

		_center_x = (_max_x + _min_x) / 2;
		_center_y = (_max_y + _min_y) / 2;
		_center_z = (_max_z + _min_z) / 2;
		_max_of_three(_max_x - _min_x, _max_y - _min_y, _max_z - _min_z, _range);

		_scale = 3 / _range;
	}

	initTransform();
}

void hGlWidget::drawTrisoup() {
	glBegin(GL_TRIANGLES);

	int i;
	for (i = 0; i < _tris_container.count(); i ++) {
		glVertex3f(_tris_container(i).vert1.x, _tris_container(i).vert1.y, _tris_container(i).vert1.z);
		glVertex3f(_tris_container(i).vert2.x, _tris_container(i).vert2.y, _tris_container(i).vert2.z);
		glVertex3f(_tris_container(i).vert3.x, _tris_container(i).vert3.y, _tris_container(i).vert3.z);
	}

	glEnd();
}

void hGlWidget::drawMCTrisoup() {
	glBegin(GL_TRIANGLES);
	HNormal nm;
	HVertex v1, v2, v3;

	int i;
	for (i = 0; i < _mc_tris.size(); i ++) {
		v1.Set(_mc_tris[i].p[0].x, _mc_tris[i].p[0].y, _mc_tris[i].p[0].z);
		v2.Set(_mc_tris[i].p[1].x, _mc_tris[i].p[1].y, _mc_tris[i].p[1].z);
		v3.Set(_mc_tris[i].p[2].x, _mc_tris[i].p[2].y, _mc_tris[i].p[2].z);
		nm = triangleNormal(v1, v2, v3);

		glNormal3f(nm.x, nm.y, nm.z);
		glVertex3f(v1.x, v1.y, v1.z);
		glVertex3f(v2.x, v2.y, v2.z);
		glVertex3f(v3.x, v3.y, v3.z);
	}

	glEnd();
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
			if (_primitive_mode == FLAT || _primitive_mode == FLAT_LINES)
				glNormal3f(fnormals[i].x, fnormals[i].y, fnormals[i].z);

			if (_primitive_mode == SMOOTH) {
				HNormal &nm = vnormals[flist[i].verts[0]];
				glNormal3f(nm.x, nm.y, nm.z);
			}
			v = vlist[f.verts[0]];
			glVertex3f(v.x, v.y, v.z);

			if (_primitive_mode == SMOOTH) {
				HNormal &nm = vnormals[flist[i].verts[2]];
				glNormal3f(nm.x, nm.y, nm.z);
			}
			v = vlist[f.verts[2]];
			glVertex3f(v.x, v.y, v.z);

			if (_primitive_mode == SMOOTH) {
				HNormal &nm = vnormals[flist[i].verts[1]];
				glNormal3f(nm.x, nm.y, nm.z);
			}
			v = vlist[f.verts[1]];
			glVertex3f(v.x, v.y, v.z);
		}
		else {
			cerr << "#error! non-triangle while drawing ply models" << endl;
		}
	}

	glEnd();
}

void hGlWidget::drawIndexedMesh() {
	glBegin(GL_TRIANGLES);
	for(int i = 0; i < numface; i ++) {
		HFace &f = faceVec[i];

		// draw front face
		if (_primitive_mode == FLAT || _primitive_mode == FLAT_LINES)
			glNormal3f(fnormals[i].x, fnormals[i].y, fnormals[i].z);

		if (_primitive_mode == SMOOTH) {
			HNormal &nm = vnormals[f.i];
			glNormal3f(nm.x, nm.y, nm.z);
		}
		HVertex &v1 = vertVec[f.i];
		glVertex3f(v1.x, v1.y, v1.z);

		if (_primitive_mode == SMOOTH) {
			HNormal &nm = vnormals[f.j];
			glNormal3f(nm.x, nm.y, nm.z);
		}
		HVertex &v2 = vertVec[f.j];
		glVertex3f(v2.x, v2.y, v2.z);

		if (_primitive_mode == SMOOTH) {
			HNormal &nm = vnormals[f.k];
			glNormal3f(nm.x, nm.y, nm.z);
		}
		HVertex &v3 = vertVec[f.k];
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

void hGlWidget::initTransform() {

	memset (_glmat, 0, sizeof (_glmat));
	_glmat[0] = _glmat[5] = _glmat[10] = _glmat[15] = 1;
	_scale = 10.0;
	_operateMode = OPERATEMODE_NONE;
	_trans_point.setValue(0.0, 0.0, 0.0);
	double _rotate_degree = 0.0;
}

void hGlWidget::applyTransform() {

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -5.0);
	//glTranslatef(_trans_point.x, _trans_point.y, _trans_point.z);
	glScalef(_scale, _scale, _scale);
	// rotate
	glMultMatrixd(_glmat);
	// locate model to (0, 0, 0)
	glTranslatef(-_center_x, -_center_y, -_center_z);
}

void hGlWidget::setLights() {
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Lights properties
	float ambientProperties[]	= {0.7f, 0.7f, 0.7f, 1.0f};
	float diffuseProperties[]	= {0.8f, 0.8f, 0.8f, 1.0f};
	float specularProperties[]	= {1.0f, 1.0f, 1.0f, 1.0f};
	float lightPosition[]		= {0.1f, 0.3f, -0.3f, 0.0f};

	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientProperties);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseProperties);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specularProperties);
	glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
	glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

	// Default : lighting
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
}

void hGlWidget::setMaterial() {

	// material
	//float	MatAmbient[]  = {0.0f, 0.5f, 0.75f, 1.0f};

	//float	MatAmbient[]  = {0.2f, 0.65f, 0.85f, 1.0f};
	//float	MatDiffuse[]  = {0.2f, 0.65f, 0.85f, 1.0f};

	float	MatAmbient[]  = {0.6f, 0.6f, 0.6f, 1.0f};
	float	MatDiffuse[]  = {0.6f, 0.6f, 0.6f, 1.0f};

	//float	MatSpecular[]  = {0.75f, 0.75f, 0.75f, 1.0f};
	float	MatSpecular[]  = {0.8f, 0.8f, 0.8, 1.0f};
	float	MatShininess[]  = { 64 };
	float	MatEmission[]  = {0.0f, 0.0f, 0.0f, 1.0f};

	//glMaterialfv(GL_FRONT, GL_AMBIENT, MatAmbient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MatDiffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, MatSpecular);
	glMaterialfv(GL_FRONT, GL_SHININESS, MatShininess);
	glMaterialfv(GL_FRONT, GL_EMISSION, MatEmission);
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
	int i = 0;
	HNormal e1, e2, nm;

	fnormals.resize(numface);
	vnormals.resize(numvert);
	for (i = 0; i < numvert; i ++)
		vnormals[i].Set(0, 0, 0);

	for (i = 0; i < numface; i ++) {
		HFace &f = faceVec[i];
		HVertex &v1 = vertVec[f.i];
		HVertex &v2 = vertVec[f.j];
		HVertex &v3 = vertVec[f.k];

		e1 = v3 - v1;
		e2 = v2 - v1;
		nm = e1 ^ e2;

		vnormals[f.i] += nm;
		vnormals[f.j] += nm;
		vnormals[f.k] += nm;

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

	_scale = 3 / _range;
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