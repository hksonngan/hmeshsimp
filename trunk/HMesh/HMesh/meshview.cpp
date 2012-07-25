#include "meshview.h"
#include "icesop_common.h"
#include "common.h"

#include "../../../Include/glut.h"

#include <cstdio>	// using vsprintf
#include <gl\gl.h>
#include <gl\glu.h>
#include <fstream>
#include <cmath>

namespace icesop {

/// matrix tools
void glmat2mat(double glmat[16], Matrix4 &mat)
{
	double *pmat = &mat.mat[0][0];

	for(int i=0; i<16; i++)
		pmat[i] = glmat[i];
}

void mat2glmat(const Matrix4 &mat, double glmat[16])
{
	const double *pmat = &mat.mat[0][0];

	for(int i=0; i<16; i++)
		glmat[i] = pmat[i];
}

MeshView::MeshView(QWidget* parent, const QString& title)
	: QGLWidget(parent)
	, operateMode_(OPERATEMODE_SELECT)
	, zoomMode_(ZOOMMODE_NONE)
	, lButtonPressPos_(QPoint(0, 0))
	, rButtonPressPos_(QPoint(0, 0))
	, currentButtonPos_(QPoint(0, 0))
	, mesh_(0)
	, viewstatus_(Status_ShadedEdge)
	, viewmode_(Mesh_vNormal)
	, zInit_(-2.414)
	, zNear_(0.1)
	, zFar_(10.0)
	, range_(1.0)
	, scale_(1.0)
	, aspect_(1.5)
{
	memset (glmat_, 0, sizeof (glmat_));
	glmat_[0] = glmat_[5] = glmat_[10] = glmat_[15] = 1;

	setMinimumSize(640, 400);
	resize(800, 600);
	setWindowTitle(title);
	setAttribute(Qt::WA_DeleteOnClose);
}

MeshView::MeshView(const QString& filename, QWidget* parent, const QString& title)
	: QGLWidget(parent)
	, operateMode_(OPERATEMODE_SELECT)
	, zoomMode_(ZOOMMODE_NONE)
	, lButtonPressPos_(QPoint(0, 0))
	, rButtonPressPos_(QPoint(0, 0))
	, currentButtonPos_(QPoint(0, 0))
	, mesh_(0)
	, viewstatus_(Status_ShadedEdge)
	, viewmode_(Mesh_vNormal)
	, zInit_(-2.414)
	, zNear_(0.1)
	, zFar_(10.0)
	, range_(1.0)
	, scale_(1.0)
	, aspect_(1.5)
{
	memset (glmat_, 0, sizeof (glmat_));
	glmat_[0] = glmat_[5] = glmat_[10] = glmat_[15] = 1;

	setCurrentFile(filename);

	setMinimumSize(640, 400);
	resize(800, 600);
	setWindowTitle(title);
	setAttribute(Qt::WA_DeleteOnClose);
}

MeshView::~MeshView()
{
	if(mesh_)
	{
		delete mesh_;
		mesh_ = 0;
	}
}

int MeshView::getMdiType()
{
	return MDITYPE_MESH;
}

void MeshView::onSetOperateMode(int mode)
{
	operateMode_ = mode;
}

void MeshView::onSetZoomMode(int mode)
{
	zoomMode_ = mode;
}

bool MeshView::onOpen()
{
	std::string tmpFileName(getCurrentFile().toLocal8Bit().data());
	std::ifstream infile(tmpFileName.c_str());
	if(!infile)
	{
		mesh_ = 0;
		return false;
	}

	int NodeNum;
	int ElementNum;
	int faceNum;
	std::string filetype;

	infile >> filetype;
	infile >> NodeNum >> ElementNum >> faceNum;

	if(mesh_)
	{
		delete mesh_;
		mesh_ = 0;
	}
	mesh_ = new Mesh();
	for(int nodeIndex = 0; nodeIndex < NodeNum; ++nodeIndex)
	{
		double x,y,z;

		Mesh::Node node(3);
		infile >> x >> y >> z;
		node.SetLocationValue(x, 0);
		node.SetLocationValue(y, 1);
		node.SetLocationValue(z, 2);
		mesh_->AddNode(node);
	}
	for(int eleIndex = 0; eleIndex < ElementNum; ++eleIndex)
	{
		int ElementNodeNum;
		infile >> ElementNodeNum;
		int nodeI1, nodeI2, nodeI3;

		Mesh::Element element(303);

		infile >> nodeI1 >> nodeI2 >> nodeI3;
		element.SetNodeIndex(nodeI1,0);
		element.SetNodeIndex(nodeI2,1);
		element.SetNodeIndex(nodeI3,2);
		mesh_->AddElement(element);
	}

	// 获取网格模型包围盒
	Mesh::Node minNode(3);
	Mesh::Node maxNode(3);
	mesh_->GetBounds(minNode, maxNode);
	Point3D minPoint(minNode.GetLocationValue(0), minNode.GetLocationValue(1), minNode.GetLocationValue(2));
	Point3D maxPoint(maxNode.GetLocationValue(0), maxNode.GetLocationValue(1), maxNode.GetLocationValue(2));

	// 设置meshDrawer包围盒
	meshDrawer_.setBounds(minPoint, maxPoint);

	range_ = max(Vector3D(minPoint.x, minPoint.y, minPoint.z).Length(), Vector3D(maxPoint.x, maxPoint.y, maxPoint.z).Length());
	zInit_ = - range_ / tan(PI / 8);
	zNear_ = range_ * 0.01;
	zFar_ = range_ * 100;
	Matrix4 mat;
	glmat2mat(glmat_, mat);
	mat.SetTranslate(0, 0, zInit_);
	mat2glmat(mat, glmat_);

	return true;
}

bool MeshView::onSave()
{
	return true;
}

bool MeshView::onSaveAs()
{
	return true;
}

void MeshView::initializeGL()
{
	// Default mode
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glShadeModel(GL_FLAT);
	glEnable(GL_NORMALIZE);

	// Lights properties
	float ambientProperties[]  = {0.7f, 0.7f, 0.7f, 1.0f};
	float diffuseProperties[]  = {0.8f, 0.8f, 0.8f, 1.0f};
	float specularProperties[] = {1.0f, 1.0f, 1.0f, 1.0f};

	glLightfv( GL_LIGHT0, GL_AMBIENT, ambientProperties);
	glLightfv( GL_LIGHT0, GL_DIFFUSE, diffuseProperties);
	glLightfv( GL_LIGHT0, GL_SPECULAR, specularProperties);
	glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0);

	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	//glClearColor(0, 0, 0, 1.0f);
	glClearColor(0.8, 0.8, 0.8, 1.0f);
	glClearDepth(1.0f);

	// Default : lighting
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);

	// Default : blending
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);

	// Default : material
	float	MatAmbient[]  = {0.0f, 0.5f, 0.75f, 1.0f};
	float	MatDiffuse[]  = {0.0f, 0.5f, 1.0f, 1.0f};
	float	MatSpecular[]  = {0.75f, 0.75f, 0.75f, 1.0f};
	float	MatShininess[]  = { 64 };
	float	MatEmission[]  = {0.0f, 0.0f, 0.0f, 1.0f};

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, MatAmbient);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, MatDiffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, MatSpecular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, MatShininess);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, MatEmission);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
}

void MeshView::paintGL()
{
	//glClearColor(0, 0, 0, 1.0f);
	glClearColor(0.8, 0.8, 0.8, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(!mesh_) return;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	if(viewmode_ == Mesh_vPerspective)
	{
		double cy = range_ / scale_ * zNear_ / zInit_;
		double cx = cy * aspect_;
		glFrustum(-cx, cx, -cy, cy, -zNear_, zFar_);
	}
	else
	{
		double cy = range_ / scale_;
		double cx = cy * aspect_;
		glOrtho(-cx, cx, -cy, cy, -zNear_, zFar_);
	}

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(glmat_);

	meshDrawer_.drawMesh(mesh_);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	paintCoord();

	glFinish();
}

void MeshView::resizeGL(int width, int height)
{
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);
	aspect_ = (height == 0) ? static_cast<double> (width) : static_cast<double> (width)/static_cast<double> (height);
	glViewport(0, 0, width, height);
	glDrawBuffer(GL_BACK);
}


void MeshView::mouseDoubleClickEvent(QMouseEvent * event)
{
	if(windowState() & Qt::WindowFullScreen)
		showNormal();
	else
		showFullScreen();
}

void MeshView::mousePressEvent(QMouseEvent * event)
{
	currentButtonPos_ = event->pos();
	if(event->buttons() == Qt::LeftButton)
	{
		lButtonPressPos_ = event->pos();

		switch(operateMode_)
		{
			case OPERATEMODE_NONE:
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
		rButtonPressPos_ = event->pos();

		switch(operateMode_)
		{
			case OPERATEMODE_NONE:
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

void MeshView::mouseReleaseEvent(QMouseEvent * event)
{
	switch(operateMode_)
	{
		case OPERATEMODE_NONE:
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

void MeshView::mouseMoveEvent(QMouseEvent * event)
{
	currentButtonPos_ = event->pos();

	switch(operateMode_)
	{
		case OPERATEMODE_NONE:
			break;
		case OPERATEMODE_SELECT:
			break;
		case OPERATEMODE_PAN:
			if(event->buttons() == Qt::LeftButton)
			{
				GLint viewport[4];
				glGetIntegerv(GL_VIEWPORT, viewport);

				double cx, cy;
				if(viewmode_ == Mesh_vPerspective)
				{
					cy = range_ / scale_ * (-glmat_[14]) / zInit_;
					cx = cy * aspect_;
				}
				else
				{
					cy = range_ / scale_;
					cx = cy * aspect_;
				}

				Matrix4 mat;
				glmat2mat(glmat_, mat);
				Matrix4 mat2 = ! mat;
        		
				Vector3D vt((currentButtonPos_.x() - lButtonPressPos_.x()) * cx * 2 / viewport[2], (- currentButtonPos_.y() + lButtonPressPos_.y()) * cy * 2 / viewport[3], 0);
				vt = vt * mat2;

				mat2.SetTranslate(vt.x, vt.y, vt.z);

				mat = mat2 * mat;
				mat2glmat(mat, glmat_);

				lButtonPressPos_ = currentButtonPos_;
				update();
			}
			break;
		case OPERATEMODE_ROTATE:
			if(event->buttons() == Qt::LeftButton)
			{
				Matrix4 mat;
				glmat2mat(glmat_, mat);
				Matrix4 mat2 = ! mat;
        		
				Vector3D vr(currentButtonPos_.x() - lButtonPressPos_.x(), - currentButtonPos_.y() + lButtonPressPos_.y(), 0);
				vr = Vector3D(0, 0, 1) ^ vr;
				vr = vr * mat2;

				double rotateSpeed = 1.0 / 3.0;
				double ang = (vr.Length() * rotateSpeed) * PI / 180;
				vr.SetUnit();
				mat2.SetRotateV(vr, ang);

				mat = mat2 * mat;
				mat2glmat(mat, glmat_);

				lButtonPressPos_ = currentButtonPos_;
				update();
			}
			break;
		case OPERATEMODE_ZOOM:
			if(event->buttons() == Qt::LeftButton)
			{
				double scaleSpeed = 1.0 / 300.0;
				double scale = 1.0 + (- currentButtonPos_.y() + lButtonPressPos_.y()) * scaleSpeed;
				scale_ *= scale;
				lButtonPressPos_ = currentButtonPos_;
				update();
			}
			break;
		default:
			break; 
	}

}

void MeshView::wheelEvent(QWheelEvent * event)
{
	double scaleSpeed = 1.0 / 300.0;
	double scale = 1.0 + (event->delta() / 5) * scaleSpeed;
	scale_ *= scale;

	update();
}

void MeshView::keyPressEvent(QKeyEvent * event)
{
}

void MeshView::keyReleaseEvent(QKeyEvent * event) 
{

}

// 绘制背景
void MeshView::paintBkgnd()
{

}

void MeshView::print_bitmap_string(void* font,const char* s)
{
	if (s && strlen(s))
	{
		while (*s)
		{
			glutBitmapCharacter(font, *s);
			++s;
		}
	}
}

// 绘制字符串
void MeshView::glPrint(const char *fmt, ...)
{
	void* bitmap_fonts[7] = {
		GLUT_BITMAP_9_BY_15,
		GLUT_BITMAP_8_BY_13,
		GLUT_BITMAP_TIMES_ROMAN_10,
		GLUT_BITMAP_TIMES_ROMAN_24,
		GLUT_BITMAP_HELVETICA_10,
		GLUT_BITMAP_HELVETICA_12,
		GLUT_BITMAP_HELVETICA_18
	};
	print_bitmap_string(bitmap_fonts[1], fmt);

}


// 绘制坐标系
void MeshView::paintCoord()
{
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	double cy = range_ / scale_;
	double cx = cy * aspect_;
	glOrtho(-cx, cx, -cy, cy, -zNear_, zFar_);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(glmat_);

	glEnable(GL_LIGHTING);

	double radius = (cx + cy) / 2 * 0.01;
	double radius2 = radius * 2;
	double length = radius * 10;
	double glmat[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, glmat);
	glmat[12] = - cx + length * 1.5;
	glmat[13] = - cy + length * 1.5;
	glmat[14] = - zNear_ - length * 1.5;
	glLoadMatrixd(glmat);

	GLfloat mat_ambient[] = { 0.3f, 0.3f, 0.3f, 1.0f };
	GLfloat mat_emission[]  = {0.0f, 0.0f, 0.0f, 1.0f};

	GLfloat mat_diffuse1[] = { 0.8f, 0.0f, 0.0f, 1.0f };
	GLfloat mat_diffuse2[] = { 0.0f, 0.8f, 0.0f, 1.0f };
	GLfloat mat_diffuse3[] = { 0.0f, 0.0f, 0.8f, 1.0f };

	GLfloat mat_specular1[] = {0.8f, 0.3f, 0.3f, 1.0f};
	GLfloat mat_specular2[] = {0.3f, 0.8f, 0.3f, 1.0f};
	GLfloat mat_specular3[] = {0.3f, 0.3f, 0.8f, 1.0f};

	GLfloat mat_shiness = 40.0;

	glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient );
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, mat_emission);
	glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, mat_shiness );

	GLUquadricObj * disk = gluNewQuadric();

	gluSphere(disk, radius, 32, 32);	// 原点

	glPushMatrix();	// Z 轴, 红色
	glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse1 );
	glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular1 );

	gluCylinder( disk, radius, radius, length, 16, 1);
	glTranslated( 0.0, 0.0, length );
	gluCylinder( disk, radius2, 0, length*0.2, 16, 1);
	glPopMatrix();

	glPushMatrix();	// Y 轴， 绿色
	glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse2 );
	glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular2 );

	glRotated(-90.0, 1.0, 0.0, 0.0);
	gluCylinder( disk, radius, radius, length, 16, 1);
	glTranslated( 0.0, 0.0, length );
	gluCylinder( disk, radius2, 0, length*0.2, 16, 1);
	glPopMatrix();

	glPushMatrix();	// X 轴，蓝色
	glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse3 );
	glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular3 );

	glRotated(90.0, 0.0, 1.0, 0.0);
	gluCylinder( disk, radius, radius, length, 16, 1);
	glTranslated( 0.0, 0.0, length );
	gluCylinder( disk, radius2, 0, length*0.2, 16, 1);
	glPopMatrix();

	gluDeleteQuadric( disk );

	glDisable(GL_LIGHTING);

	glColor3f(0.8f, 0.0f, 0.0f);
	glRasterPos3d(-radius2, -radius2, length*1.2);
	glPrint("Z");

	glColor3f(0.0f, 0.8f, 0.0f) ;
	glRasterPos3d(-radius2, length*1.2, -radius2);
	glPrint("Y");

	glColor3f(0.0f, 0.0f, 0.8f) ;
	glRasterPos3d(length*1.2, -radius2, -radius2);
	glPrint("X");

	glEnable(GL_LIGHTING);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}


void MeshView::SetScale(double scale)
{
	if(scale>0)
		scale_ = scale;
}



} // namespace icesop
