#include "hGlWidget.h"

#include <algorithm>

#include "common.h"
#include "icesop_common.h"
#include "ply/ply_inc.h"
#include "tri_soup_stream.h"

#define _max_of_three(a, b, c, _max)  _max = max(a, b); _max = max(c, _max);

hGlWidget::hGlWidget()
{
	_draw_qslim = false;
	_draw_ply = false;
	_draw_tris = false;

	_primitive_mode = FLAT_LINES;
	_color_mode = FACE_COLOR;

	fnormals = NULL;
	vnormals = NULL;

	memset (_glmat, 0, sizeof (_glmat));
	_glmat[0] = _glmat[5] = _glmat[10] = _glmat[15] = 1;
	_scale = 10.0;
	_operateMode = OPERATEMODE_NONE;
	_trans_point.setValue(0.0, 0.0, 0.0);
	double _rotate_degree = 0.0;
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

void hGlWidget::initializeGL()
{
	//glShadeModel(GL_FLAT);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_NORMALIZE);

	// Default : blending
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);

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

	if(_draw_ply)
	{
		//if (_primitive_mode == FLAT_LINES || _primitive_mode == WIREFRAME) {
		//	glDisable(GL_LIGHTING);
		//	glColor3f(0.0f, 0.0f, 0.0f);
		//	glPolygonMode(GL_FRONT, GL_LINE);
		//	drawModel();
		//}
		
		if (_primitive_mode != WIREFRAME) {
			setLights();
			applyTransform();
			setMaterial();
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			drawModel();	
		}
	}

	if (_draw_tris)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDisable(GL_LIGHTING);
		glColor3f(0.1f, 0.1f, 0.1f);
		glBegin(GL_TRIANGLES);

		int i;
		for (i = 0; i < _tris_container.count(); i ++)
		{
			glVertex3f(_tris_container(i).vert1.x, _tris_container(i).vert1.y, _tris_container(i).vert1.z);
			glVertex3f(_tris_container(i).vert2.x, _tris_container(i).vert2.y, _tris_container(i).vert2.z);
			glVertex3f(_tris_container(i).vert3.x, _tris_container(i).vert3.y, _tris_container(i).vert3.z);
		}

		glEnd();
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
	_draw_qslim = true;
	_draw_ply = false;
	_draw_tris = false;
	update();
}

void hGlWidget::setDrawPly()
{
	_draw_ply = true;
	_draw_qslim = false;
	_draw_tris = false;
	update();
}

void hGlWidget::setDrawTris()
{
	_draw_tris = true;
	_draw_ply = false;
	_draw_qslim = false;
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
	double scaleSpeed = 1.0 / 300.0;
	double scale = 1.0 + (- event->delta() / 5) * scaleSpeed;
	_scale *= scale;

	update();
}

void hGlWidget::openFile(QString _file_name)
{
	QString _file_ext = _file_name.mid(_file_name.lastIndexOf(".") + 1);

	if (_file_ext.toLower() == "ply")
	{
		clean_ply();
		_tris_container.clear();
		ply_read_file(_file_name.toLocal8Bit().data());
		computeNormals();

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
		calcBoundingBox();
	}
	else if (_file_ext.toLower() == "tris")
	{
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
}

void hGlWidget::drawModel() {

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

void hGlWidget::applyTransform() {

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -5.0);
	glTranslatef(_trans_point.x, _trans_point.y, _trans_point.z);
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
	float lightPosition[]		= {0.0f, 0.0f, -0.3f, 0.0f};

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
	float	MatAmbient[]  = {0.1f, 0.15f, 0.35f, 1.0f};
	float	MatDiffuse[]  = {0.1f, 0.3f, 0.35f, 1.0f};
	//float	MatSpecular[]  = {0.75f, 0.75f, 0.75f, 1.0f};
	float	MatSpecular[]  = {0.0f, 0.0f, 0.0f, 1.0f};
	float	MatShininess[]  = { 64 };
	float	MatEmission[]  = {0.0f, 0.0f, 0.0f, 1.0f};

	glMaterialfv(GL_FRONT, GL_AMBIENT, MatAmbient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, MatDiffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, MatSpecular);
	glMaterialfv(GL_FRONT, GL_SHININESS, MatShininess);
	glMaterialfv(GL_FRONT, GL_EMISSION, MatEmission);
}
 
void hGlWidget::computeNormals() {

	int i = 0;
	HVertex v1, v2, v3;
	HNormal e1, e2, nm;

	if (vnormals)
		delete[] vnormals;
	if (fnormals)
		delete[] fnormals;

	fnormals = new HNormal[nfaces];

	vnormals = new HNormal[nverts];
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

void hGlWidget::calcBoundingBox()
{
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

	_center_x = (_max_x + _min_x) / 2;
	_center_y = (_max_y + _min_y) / 2;
	_center_z = (_max_z + _min_z) / 2;
	_max_of_three(_max_x - _min_x, _max_y - _min_y, _max_z - _min_z, _range);

	_scale = 3 / _range;
}