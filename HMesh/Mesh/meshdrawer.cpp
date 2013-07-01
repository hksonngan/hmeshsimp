#include "MeshDrawer.h"

#include "GL/GLU.h"
#include <cmath>
#include <fstream>

namespace icesop {

static void normalize(double v[3]) {    
	GLdouble d = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); 
	if (d == 0.0) 	return;
	d = 1/d;
	v[0] *= d; v[1] *= d; v[2] *= d; 
}

static void normcrossprod(double v1[3], double v2[3], double out[3]) 
{ 
	out[0] = v1[1]*v2[2] - v1[2]*v2[1]; 
	out[1] = v1[2]*v2[0] - v1[0]*v2[2]; 
	out[2] = v1[0]*v2[1] - v1[1]*v2[0]; 
	normalize(out); 
}

static double dotprod(double v1[3], double v2[3])
{
	return v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
}

MeshDrawer::MeshDrawer()
	: mesh_(0)
	, bounds_(Bounds(Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, 0.0)))
	, cuttingPlaneA_(1.0), cuttingPlaneB_(0.0), cuttingPlaneC_(0.0), cuttingPlaneD_(0.0)
	, colorR_(1), colorG_(1), colorB_(1), lineAlpha_(1.0)
	, pickedR_(1.0), pickedG_(0.0), pickedB_(1.0)
	, lineWidth_(1.0)
{
}

MeshDrawer::MeshDrawer(Mesh* mesh)
	: mesh_(mesh)
	, bounds_(Bounds(Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, 0.0)))
	, cuttingPlaneA_(1.0), cuttingPlaneB_(0.0), cuttingPlaneC_(0.0), cuttingPlaneD_(0.0)
	, colorR_(1), colorG_(1), colorB_(1), lineAlpha_(1.0)
	, pickedR_(1.0), pickedG_(0.0), pickedB_(1.0)
	, lineWidth_(3.0)
{
}

MeshDrawer::MeshDrawer(const std::string& filename)
	: bounds_(Bounds(Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, 0.0)))
	, cuttingPlaneA_(1.0), cuttingPlaneB_(0.0), cuttingPlaneC_(0.0), cuttingPlaneD_(0.0)
	, colorR_(1), colorG_(1), colorB_(1), lineAlpha_(1.0)
	, pickedR_(1.0), pickedG_(0.0), pickedB_(1.0)
	, lineWidth_(1.0)
{
	std::ifstream infile(filename.c_str());
	if(!infile)
	{
		mesh_ = 0;
		return;
	}

	int NodeNum;
	int ElementNum;
	int faceNum;
	std::string filetype;

	infile >> filetype;
	infile >> NodeNum >> ElementNum >> faceNum;

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
}

MeshDrawer::~MeshDrawer()
{
}

void MeshDrawer::setMesh(Mesh* mesh)
{
	mesh_ = mesh;
}

void MeshDrawer::setMesh(const std::string& filename)
{
}

Mesh* MeshDrawer::getMesh()
{
	return mesh_;
}

// ����ѡ�񼯺�
void MeshDrawer::setCurrentSelectedSet(GeometrySet* set)
{
	currentSelectedSet_ = set;
}

// ��ȡѡ�񼯺�
GeometrySet* MeshDrawer::getCurrentSelectedSet()
{
	return currentSelectedSet_;
}

// ���ñ�ѡ��ʱ����ɫ 0.0 < r,g,b < 1.0
void MeshDrawer::setPickedColor(float r, float g, float b)
{
	pickedR_ = r;
	pickedG_ = g;
	pickedB_ = b;
}

// ����������ɫ  0.0 < r,g,b < 1.0
void MeshDrawer::setColor(float r, float g, float b, float alpha)
{
	colorR_ = r;
	colorG_ = g;
	colorB_ = b;
	lineAlpha_ = alpha;
}

// ����������� 
void MeshDrawer::setLineWidth(float w)
{
	lineWidth_ = w;
}

// �������������
void MeshDrawer::setPlane(double a, double b, double c, double d)
{
	cuttingPlaneA_ = a;
	cuttingPlaneB_ = b;
	cuttingPlaneC_ = c;
	cuttingPlaneD_ = d;
}

// ���ð�Χ��
void MeshDrawer::setBounds(const Bounds& bounds)
{
	bounds_ = bounds;
}

void MeshDrawer::setBounds(const Point3D& pmin, const Point3D& pmax)
{
	bounds_ = Bounds(pmin, pmax);
}

// ��ȡ��Χ��
Bounds MeshDrawer::getBounds()
{
	return bounds_;
}


// ���Ƶ�����
inline void glNode(const Mesh::Node& node)
{
	glVertex3d(node.GetLocationValue(0),node.GetLocationValue(1),node.GetLocationValue(2));
}


bool MeshDrawer::isFaceSelected(int face, GeometrySet* selectedSet) const
{
	int numSelectedFaces = selectedSet->GetFacesNum();
	for (int i = 0; i < numSelectedFaces; i++)
	{
		if (face == selectedSet->GetFaceIndex(i))
			return true;
	}
	return false;
}

bool MeshDrawer::isVertexSelected(int vertex, GeometrySet* selectedSet) const
{
	int numSelectedVertices = selectedSet->GetVerticesNum();
	for (int i = 0; i < numSelectedVertices; i++)
	{
		if (vertex == selectedSet->GetVertexIndex(i))
			return true;
	}
	return false;
}

bool MeshDrawer::isEdgeSelected(int edge, GeometrySet* selectedSet) const
{
	int numSelectedEdges = selectedSet->GetEdgesNum();
	for (int i = 0; i < numSelectedEdges; i++)
	{
		if (edge == selectedSet->GetEdgeIndex(i))
			return true;
	}
	return false;
}


void getPerpendicularVector(double a, double b, double c, double& x, double& y, double& z)
{
	if ( a != 0)
	{
		x = -(b+c)/a;
		y = 1;
		z = 1;
	}
	else if ( b != 0 )
	{
		y = -(a+c)/b;
		x = 1;
		z = 1;
	}
	else if ( c != 0 )
	{
		z = -(a+b)/c;
		x = 1;
		y = 1;
	}
}

// ���mesh����ƽ�� ax + by +cz = d �ཻ��element��ţ����� std::vector<Gluint> hitElements��
// ��OpenGL��Pickingʵ�֣��ŵ���ƽ��Ĳ������⣬����ʵ����תʱ���á�Ŀǰ���д�...������
void MeshDrawer::getHitElements(const Mesh* mesh, std::vector<GLuint>& hitElements, double a, double b, double c, double d)
{
	// vector that perpendicular to (a,b,c)
	double x,y,z;
	getPerpendicularVector(a,b,c,x,y,z);
	double distanceOriginToPlane;
	distanceOriginToPlane = d / sqrt( a*a + b*b + c*c );

	static GLuint* selectBuffer = new GLuint[0x1ffff];
	glSelectBuffer(0x1ffff, selectBuffer);


	glRenderMode (GL_SELECT);

	const double enoughRadius = sqrt( pow( (bounds_.getURB().x - bounds_.getLLF().x), 2 ) +
									  pow( (bounds_.getURB().y - bounds_.getLLF().y), 2 ) +
									  pow( (bounds_.getURB().z - bounds_.getLLF().z), 2 ) );
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho( -enoughRadius, enoughRadius, 1E-10, -1E-10, -enoughRadius, enoughRadius);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	// ����ģ��λ�ã�ʹ��ƽ�� ax+by+cz=d �� xz ƽ���غ�
	glTranslated(0,-distanceOriginToPlane,0);
	gluLookAt(0,0,0, x,y,z, a,b,c);

	glInitNames();
	glPushName(0);
	int elementsNumber = mesh->GetElementsNumber();
	for(int i = 0; i < elementsNumber; i++)
	{
		const Mesh::Element & currentElement = mesh->GetConstElement(i);

		const Mesh::Node& node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));
		const Mesh::Node& node3 = mesh->GetConstNode(currentElement.GetNodeIndex(3));
		glLoadName(i);
		glBegin(GL_LINE_LOOP);
			glNode(node0);
			glNode(node1);
			glNode(node2);
			glNode(node3);
		glEnd();
	}

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	GLuint hits = glRenderMode(GL_RENDER);
	GLuint* ptr = selectBuffer + 3;
	for (int i = 0; i < hits; i++, ptr += 4)
	{
		hitElements.push_back(*ptr);
	}
}

// �������ı���ʵ�֣�Ŀǰ��֧�ַ�����Ϊx,y,z�����ƽ��
void MeshDrawer::getCutElements(const Mesh* mesh, std::vector<GLuint>& cutElements, double a, double b, double c, double d)
{
	double cutValue;
	int direction;
	if( a != 0)
	{
		cutValue = d / a;
		direction = 0;
	}
	else if( b != 0)
	{
		cutValue = d / b;
		direction = 1;
	}
	else if( c != 0)
	{
		cutValue = d / c;
		direction = 2;
	}
	for( int i = 0; i < mesh->GetElementsNumber(); ++i)
	{
		const Mesh::Element & currentElement = mesh->GetConstElement(i);

		double value0 = mesh->GetConstNode(currentElement.GetNodeIndex(0)).GetLocationValue(direction);
		double value1 = mesh->GetConstNode(currentElement.GetNodeIndex(1)).GetLocationValue(direction);
		double value2 = mesh->GetConstNode(currentElement.GetNodeIndex(2)).GetLocationValue(direction);
		double value3 = mesh->GetConstNode(currentElement.GetNodeIndex(3)).GetLocationValue(direction);
		if( value0 <= cutValue && value1 <= cutValue && value2 <= cutValue && value3 <= cutValue
			|| value0 >= cutValue && value1 >= cutValue && value2 >= cutValue && value3 >= cutValue )
			continue;
		else
			cutElements.push_back(i);

	}
}

// ��ȡ����������ʱӦ���Ƶı�����������
void MeshDrawer::getShowSurfaceElements(const Mesh* mesh, std::vector<GLuint>& surfaceElements, double a, double b, double c, double d)
{
	double cutValue;
	int direction;
	if( a != 0)
	{
		cutValue = d / a;
		direction = 0;
	}
	else if( b != 0)
	{
		cutValue = d / b;
		direction = 1;
	}
	else if( c != 0)
	{
		cutValue = d / c;
		direction = 2;
	}
	for( int i = 0; i < mesh->GetSurfaceElementsNumber(); ++i)
	{
		const Mesh::Element & currentElement = mesh->GetConstSurfaceElement(i);

		double value0 = mesh->GetConstNode(currentElement.GetNodeIndex(0)).GetLocationValue(direction);
		double value1 = mesh->GetConstNode(currentElement.GetNodeIndex(1)).GetLocationValue(direction);
		double value2 = mesh->GetConstNode(currentElement.GetNodeIndex(2)).GetLocationValue(direction);
		if( value0 <= cutValue && value1 <= cutValue && value2 <= cutValue )
			surfaceElements.push_back(i);
	}
}

// ��������,ע��һ��Ҫ�ڻ�������֮�����
void MeshDrawer::drawCuttingPlane()
{
	glDisable(GL_LIGHTING);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	//Draw The Bounding Box
	glColor4d(1.0, 1.0, 1.0, 1.0);
	glBegin(GL_LINE_LOOP);
		glVertex3d( bounds_.getLLF().x, bounds_.getLLF().y, bounds_.getLLF().z );
		glVertex3d( bounds_.getLLF().x, bounds_.getLLF().y, bounds_.getURB().z );
		glVertex3d( bounds_.getURB().x, bounds_.getLLF().y, bounds_.getURB().z );
		glVertex3d( bounds_.getURB().x, bounds_.getLLF().y, bounds_.getLLF().z );
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3d( bounds_.getLLF().x, bounds_.getURB().y, bounds_.getLLF().z );
		glVertex3d( bounds_.getLLF().x, bounds_.getURB().y, bounds_.getURB().z );
		glVertex3d( bounds_.getURB().x, bounds_.getURB().y, bounds_.getURB().z );
		glVertex3d( bounds_.getURB().x, bounds_.getURB().y, bounds_.getLLF().z );
	glEnd();

	glBegin(GL_LINES);
		glVertex3d( bounds_.getLLF().x, bounds_.getURB().y, bounds_.getLLF().z );
		glVertex3d( bounds_.getLLF().x, bounds_.getLLF().y, bounds_.getLLF().z );
		glVertex3d( bounds_.getLLF().x, bounds_.getURB().y, bounds_.getURB().z );
		glVertex3d( bounds_.getLLF().x, bounds_.getLLF().y, bounds_.getURB().z );
		glVertex3d( bounds_.getURB().x, bounds_.getURB().y, bounds_.getURB().z );
		glVertex3d( bounds_.getURB().x, bounds_.getLLF().y, bounds_.getURB().z );
		glVertex3d( bounds_.getURB().x, bounds_.getURB().y, bounds_.getLLF().z );
		glVertex3d( bounds_.getURB().x, bounds_.getLLF().y, bounds_.getLLF().z );
	glEnd();


	const float xOffset = (bounds_.getURB().x - bounds_.getLLF().x) * 0.1;
	const float yOffset = (bounds_.getURB().y - bounds_.getLLF().y) * 0.1;
	const float zOffset = (bounds_.getURB().z - bounds_.getLLF().z) * 0.1;
	GLdouble eq0[] = { 0, 1, 0, -( bounds_.getLLF().y - yOffset )};
	GLdouble eq1[] = { 0, -1, 0,   bounds_.getURB().y + yOffset };
	GLdouble eq2[] = { 1, 0, 0, -( bounds_.getLLF().x - xOffset )};
	GLdouble eq3[] = { -1, 0, 0,   bounds_.getURB().x + xOffset };
	GLdouble eq4[] = { 0, 0, 1, -( bounds_.getLLF().z - zOffset )};
	GLdouble eq5[] = { 0, 0, -1,   bounds_.getURB().z + zOffset };
	glClipPlane(GL_CLIP_PLANE0, eq0);
	glClipPlane(GL_CLIP_PLANE1, eq1);
	glClipPlane(GL_CLIP_PLANE2, eq2);
	glClipPlane(GL_CLIP_PLANE3, eq3);
	glClipPlane(GL_CLIP_PLANE4, eq4);
	glClipPlane(GL_CLIP_PLANE5, eq5);
	glEnable(GL_CLIP_PLANE0);
	glEnable(GL_CLIP_PLANE1);
	glEnable(GL_CLIP_PLANE2);
	glEnable(GL_CLIP_PLANE3);
	glEnable(GL_CLIP_PLANE4);
	glEnable(GL_CLIP_PLANE5);

	//��ƽ�� xz ����һ��ʮ�ִ���ı��Σ�֮����ת����ƽ�� ax+by+cz=d�غ�
	//��ƽ�淨�����ļн�
	const double enoughRadius = sqrt( pow( (bounds_.getURB().x - bounds_.getLLF().x), 2 ) +
									  pow( (bounds_.getURB().y - bounds_.getLLF().y), 2 ) +
									  pow( (bounds_.getURB().z - bounds_.getLLF().z), 2 ) );
	double angle = acos( cuttingPlaneB_/ sqrt( cuttingPlaneA_*cuttingPlaneA_ + cuttingPlaneB_*cuttingPlaneB_ + cuttingPlaneC_*cuttingPlaneC_) ) * 180 / PI;
	double distanceOriginToPlane = cuttingPlaneD_ / sqrt( cuttingPlaneA_*cuttingPlaneA_ + cuttingPlaneB_*cuttingPlaneB_ + cuttingPlaneC_*cuttingPlaneC_ );
	glRotated( -angle, -cuttingPlaneC_, 0, cuttingPlaneA_ );
	glTranslated( 0, distanceOriginToPlane, 0) ;
	glColor4d( 0.8, 0.8, 0.8, 0.7);
	glBegin(GL_QUADS);
		glVertex3d( enoughRadius, 0, enoughRadius );
		glVertex3d( -enoughRadius, 0, enoughRadius );
		glVertex3d( -enoughRadius, 0, -enoughRadius );
		glVertex3d( enoughRadius, 0, -enoughRadius );
	glEnd();

	glDisable(GL_CLIP_PLANE0);
	glDisable(GL_CLIP_PLANE1);
	glDisable(GL_CLIP_PLANE2);
	glDisable(GL_CLIP_PLANE3);
	glDisable(GL_CLIP_PLANE4);
	glDisable(GL_CLIP_PLANE5);

	glPopMatrix();
	glDisable(GL_BLEND);
	glDisable(GL_LINE_SMOOTH);
}

// ���ƽڵ�
void MeshDrawer::drawNodes(const Mesh* mesh, GeometrySet* currentSelectedSet)
{
	int nodeNum = mesh->GetNodesNumber();

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT2);
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHT1);

	glPolygonOffset(1.0, 1.0);

	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	for(int i = 0; i < nodeNum; ++i)
	{
		const Mesh::Node& node = mesh->GetNode(i);
		glBegin(GL_POINTS);
			glColor3f(colorR_, colorG_, colorB_);
			glNode(node);
		glEnd();
	}
	glDisable(GL_COLOR_MATERIAL);

	return;
}

// ������������
void MeshDrawer::drawTriangleSurface(const Mesh* mesh, GeometrySet* currentSelectedSet)
{
	int edgesNumber    = mesh->GetEdgesNumber();
	int verticesNumber = mesh->GetVerticesNumber();
	int elementsNumber = mesh->GetElementsNumber();

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT2);
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHT1);

	glPolygonOffset(1.0, 1.0);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	for(int i = 0; i < elementsNumber; ++i)
	{
		//int elementFace = mesh->getFaceIndexOfElement(i);
		const Mesh::Element & currentElement = mesh->GetConstElement(i);
		const Mesh::Node& Node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& Node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& Node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));
		glBegin(GL_TRIANGLES);
			//glColor3f( colorR_, colorG_, colorB_);
			glColor3f( 0.0756800000f, 0.6142400000f, 0.0756800000f );
			glNode(Node0);
			glNode(Node1);
			glNode(Node2);
		glEnd();
	}
	glDisable(GL_COLOR_MATERIAL);	
	glDisable(GL_POLYGON_OFFSET_FILL);

	//����
	glDisable(GL_LIGHTING);
	glColor3f(colorR_, colorG_, colorB_);
	glLineWidth(lineWidth_);
	for(int i = 0; i < elementsNumber; i++)
	{
		//int elementFace = mesh->getFaceIndexOfElement(i);
		const Mesh::Element & currentElement = mesh->GetConstElement(i);
		const Mesh::Node& Node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& Node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& Node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));

		glBegin(GL_LINE_LOOP);
			glNode(Node0);
			glNode(Node1);
			glNode(Node2);
		glEnd();
	}
}

// ����������
void MeshDrawer::drawTetra(const Mesh* mesh, GeometrySet* currentSelectedSet)
{
	//int edgesNumber    = mesh->GetEdgesNumber();
	//int verticesNumber = mesh->GetVerticesNumber();
	int elementsNumber = mesh->GetElementsNumber();

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT2);
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHT1);

	glPolygonOffset(1.0,1.0);

	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	for(int i = 0; i < elementsNumber; i++)
	{
		//int elementFace = mesh->getFaceIndexOfElement(i);
		const Mesh::Element & currentElement = mesh->GetConstElement(i);
		const Mesh::Node& Node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& Node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& Node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));
		const Mesh::Node& Node3 = mesh->GetConstNode(currentElement.GetNodeIndex(3));
		glBegin(GL_TRIANGLES);
			glColor3f(colorR_, colorG_, colorB_);

			//������İڷ�˳��ͬ�������ɵ������εķ���Ҳ����ͬ���������ν�������ɫ������Ϊ��ɫ������Ϊ��ɫ
			//��ˣ��������θ������ڰڷ�ʱӦ����˳ʱ������ʱ�����⡣��������׼��
			glNode(Node0);
			glNode(Node2);
			glNode(Node1);
        	
			glNode(Node2);
			glNode(Node3);
			glNode(Node1);
        	
			glNode(Node0);
			glNode(Node1);
			glNode(Node3);
        	
			glNode(Node0);
			glNode(Node3);
			glNode(Node2);
		glEnd();
	}
	glDisable(GL_COLOR_MATERIAL);

	//����
	glDisable(GL_LIGHTING);
	glColor3f(colorR_, colorG_, colorB_);
	glLineWidth(lineWidth_);
	for(int i = 0; i < elementsNumber; i++)
	{
		const Mesh::Element & currentElement = mesh_->GetConstElement(i);
		const Mesh::Node& Node0 = mesh_->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& Node1 = mesh_->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& Node2 = mesh_->GetConstNode(currentElement.GetNodeIndex(2));
		const Mesh::Node& Node3 = mesh_->GetConstNode(currentElement.GetNodeIndex(3));

		glBegin(GL_LINE_LOOP);
			glNode(Node0);
			glNode(Node1);
			glNode(Node2);
			glNode(Node0);		
			glNode(Node3);
			glNode(Node1);
		glEnd();

		glBegin(GL_LINES);
			glNode(Node2);
			glNode(Node3);
		glEnd();
	}
}

// ����������������߿�ģ��
void MeshDrawer::drawTetraWire(const Mesh* mesh, GeometrySet* currentSelectedSet)
{
	glDisable(GL_LIGHTING);
	glColor3f(colorR_, colorG_, colorB_);
	glLineWidth(lineWidth_);
	int elementsNumber = mesh->GetElementsNumber();
	for(int i = 0; i < elementsNumber; i++)
	{
		const Mesh::Element & currentElement = mesh->GetConstElement(i);
		const Mesh::Node& node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));
		const Mesh::Node& node3 = mesh->GetConstNode(currentElement.GetNodeIndex(3));
		glBegin(GL_LINE_STRIP);
			glNode(node0);
			glNode(node1);
			glNode(node2);
			glNode(node0);
			glNode(node3);
			glNode(node2);
			glNode(node1);
			glNode(node3);
		glEnd();
	}
}

// �����������������ģ��
void MeshDrawer::drawTetraHides(const Mesh* mesh, std::vector<GLuint>& hitElements, std::vector<GLuint>& hitSurfElements,
								double a, double b, double c, double d)
{
	int hitElementsNum = hitElements.size();
	int hitSurfElementsNum = hitSurfElements.size();

	//��������hitElement��hitSurfElement
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	glColor3f(1.0, 0.0, 0.0);


	//�˴���Ҫ��������polygon offset�����������ߺͶ���ε�stitching
	glPolygonOffset(0.45, 1.0);

	glBegin(GL_TRIANGLES);
	for (int i = 0; i < hitElementsNum; i++)
	{
		const Mesh::Element & currentElement = mesh->GetConstElement(hitElements[i]);
		const Mesh::Node& node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));
		const Mesh::Node& node3 = mesh->GetConstNode(currentElement.GetNodeIndex(3));

		//���뱣֤������Ƭ��front���⣬������ʾ�᲻��ȷ(Ҳ��������Ϊû�����ú�front��back...�д����)
		glNode(node2);
		glNode(node1);
		glNode(node0);

		glNode(node2);
		glNode(node3);
		glNode(node1);

		glNode(node0);
		glNode(node3);
		glNode(node2);	

		glNode(node1);
		glNode(node3);
		glNode(node0);
	}

	glPolygonOffset(1.0, 1.0);
	GLfloat mat_diffuse1[] = { 0.8f, 0.0f, 0.0f, 1.0f };
	GLfloat mat_diffuse2[] = { 0.0f, 0.8f, 0.0f, 1.0f };
	GLfloat mat_diffuse3[] = { 0.0f, 0.0f, 0.8f, 1.0f };
	glColor3f( mat_diffuse1[0],  mat_diffuse1[1],  mat_diffuse1[2] );


	for(int i = 0; i < hitSurfElementsNum; i++)
	{
		const Mesh::Element & currentElement = mesh->GetConstSurfaceElement(hitSurfElements[i]);
		const Mesh::Node& Node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& Node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& Node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));

		glNode(Node0);	
		glNode(Node1);	
		glNode(Node2);	
	}

	glEnd();
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_LIGHTING);

	glColor3f(colorR_, colorG_, colorB_);
	glLineWidth(lineWidth_);
	for (int i = 0; i < hitElementsNum; i++)
	{
		const Mesh::Element & currentElement = mesh->GetConstElement(hitElements[i]);
		const Mesh::Node& node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));
		const Mesh::Node& node3 = mesh->GetConstNode(currentElement.GetNodeIndex(3));

		glBegin(GL_LINE_STRIP);
			glNode(node0);
			glNode(node1);
			glNode(node2);
			glNode(node0);
			glNode(node3);
			glNode(node2);
			glNode(node1);
			glNode(node3);
		glEnd();
	}

	for(int i = 0; i < hitSurfElementsNum; i++)
	{
		const Mesh::Element & currentElement = mesh->GetConstSurfaceElement(hitSurfElements[i]);
		const Mesh::Node& Node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& Node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& Node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));
		glBegin(GL_LINE_LOOP);
			glNode(Node0);
			glNode(Node1);
			glNode(Node2);
        glEnd();
	}
}

// ����������
void MeshDrawer::drawHex(const Mesh* mesh, GeometrySet* currentSelectedSet)
{
	glDisable(GL_LIGHTING);
	glColor3f(colorR_, colorG_, colorB_);
	glLineWidth(lineWidth_);
	int elementsNumber = mesh->GetSurfaceElementsNumber();
	for(int i = 0; i < elementsNumber; ++i)
	{
		const Mesh::Element & currentElement = mesh->GetConstSurfaceElement(i);
		const Mesh::Node& node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));
		const Mesh::Node& node3 = mesh->GetConstNode(currentElement.GetNodeIndex(3));
		const Mesh::Node& node4 = mesh->GetConstNode(currentElement.GetNodeIndex(4));
		const Mesh::Node& node5 = mesh->GetConstNode(currentElement.GetNodeIndex(5));
		const Mesh::Node& node6 = mesh->GetConstNode(currentElement.GetNodeIndex(6));
		const Mesh::Node& node7 = mesh->GetConstNode(currentElement.GetNodeIndex(7));
		glBegin(GL_LINE_STRIP);
			glNode(node0);
			glNode(node1);
			glNode(node2);
			glNode(node3);
			glNode(node0);
			glNode(node4);
			glNode(node5);
			glNode(node6);
			glNode(node7);
			glNode(node4);
		glEnd();
		glBegin(GL_LINES);
			glNode(node1);
			glNode(node5);
			glNode(node2);
			glNode(node6);
			glNode(node3);
			glNode(node7);
		glEnd();
	}
}

// ����������������߿�ģ��
void MeshDrawer::drawHexWire(const Mesh* mesh, GeometrySet* currentSelectedSet)
{
	glDisable(GL_LIGHTING);
	glColor3f(colorR_, colorG_, colorB_);
	glLineWidth(lineWidth_);
	int elementsNumber = mesh->GetSurfaceElementsNumber();
	for(int i = 0; i < elementsNumber; ++i)
	{
		const Mesh::Element & currentElement = mesh->GetConstSurfaceElement(i);
		const Mesh::Node& node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));
		const Mesh::Node& node3 = mesh->GetConstNode(currentElement.GetNodeIndex(3));
		const Mesh::Node& node4 = mesh->GetConstNode(currentElement.GetNodeIndex(4));
		const Mesh::Node& node5 = mesh->GetConstNode(currentElement.GetNodeIndex(5));
		const Mesh::Node& node6 = mesh->GetConstNode(currentElement.GetNodeIndex(6));
		const Mesh::Node& node7 = mesh->GetConstNode(currentElement.GetNodeIndex(7));
		glBegin(GL_LINE_STRIP);
			glNode(node0);
			glNode(node1);
			glNode(node2);
			glNode(node3);
			glNode(node0);
			glNode(node4);
			glNode(node5);
			glNode(node6);
			glNode(node7);
			glNode(node4);
		glEnd();
		glBegin(GL_LINES);
			glNode(node1);
			glNode(node5);
			glNode(node2);
			glNode(node6);
			glNode(node3);
			glNode(node7);
		glEnd();
	}
}

// �������������������ģ��
void MeshDrawer::drawHexHides(const Mesh* mesh, std::vector<GLuint>& hitElements, std::vector<GLuint>& hitSurfElements,
					  		  double a, double b, double c, double d)
{
	GLfloat mat_diffuse1[] = { 0.8f, 0.0f, 0.0f, 1.0f };
	GLfloat mat_diffuse2[] = { 0.0f, 0.8f, 0.0f, 1.0f };
	GLfloat mat_diffuse3[] = { 0.0f, 0.0f, 0.8f, 1.0f };

	glDisable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	glColor3f( mat_diffuse1[0],  mat_diffuse1[1],  mat_diffuse1[2] );

	double v[6][3], n[3];
	double positive_n[3]={0.0, 0.0, 1.0};
	double negative_n[3]={0.0, 0.0, -1.0};
	int size = mesh->GetSurfaceElementsNumber();
	glBegin(GL_QUADS);
	for (int i = 0; i < size; ++i)
	{
		const Mesh::Element& element = mesh->GetConstSurfaceElement(i);
		const Mesh::Node& node0 = mesh->GetConstNode(element.GetNodeIndex(0));
		const Mesh::Node& node1 = mesh->GetConstNode(element.GetNodeIndex(1));
		const Mesh::Node& node2 = mesh->GetConstNode(element.GetNodeIndex(2));
		const Mesh::Node& node3 = mesh->GetConstNode(element.GetNodeIndex(3));
		const Mesh::Node& node4 = mesh->GetConstNode(element.GetNodeIndex(4));
		const Mesh::Node& node5 = mesh->GetConstNode(element.GetNodeIndex(5));
		const Mesh::Node& node6 = mesh->GetConstNode(element.GetNodeIndex(6));
		const Mesh::Node& node7 = mesh->GetConstNode(element.GetNodeIndex(7));

		v[0][0] = node1.GetLocationValue(0) - node0.GetLocationValue(0);
		v[0][1] = node1.GetLocationValue(1) - node0.GetLocationValue(1);
		v[0][2] = node1.GetLocationValue(2) - node0.GetLocationValue(2);

		v[1][0] = node4.GetLocationValue(0) - node0.GetLocationValue(0);
		v[1][1] = node4.GetLocationValue(1) - node0.GetLocationValue(1);
		v[1][2] = node4.GetLocationValue(2) - node0.GetLocationValue(2);

		v[2][0] = node3.GetLocationValue(0) - node0.GetLocationValue(0);
		v[2][1] = node3.GetLocationValue(1) - node0.GetLocationValue(1);
		v[2][2] = node3.GetLocationValue(2) - node0.GetLocationValue(2);

		v[3][0] = node5.GetLocationValue(0) - node6.GetLocationValue(0);
		v[3][1] = node5.GetLocationValue(1) - node6.GetLocationValue(1);
		v[3][2] = node5.GetLocationValue(2) - node6.GetLocationValue(2);

		v[4][0] = node2.GetLocationValue(0) - node6.GetLocationValue(0);
		v[4][1] = node2.GetLocationValue(1) - node6.GetLocationValue(1);
		v[4][2] = node2.GetLocationValue(2) - node6.GetLocationValue(2);

		v[5][0] = node7.GetLocationValue(0) - node6.GetLocationValue(0);
		v[5][1] = node7.GetLocationValue(1) - node6.GetLocationValue(1);
		v[5][2] = node7.GetLocationValue(2) - node6.GetLocationValue(2);

		normcrossprod(v[0], v[2], n);
		glNode(node0);
		glNode(node3);
		glNode(node2);
		glNode(node1);

		normcrossprod(v[5], v[3], n);
		glNode(node4);
		glNode(node5);
		glNode(node6);
		glNode(node7);
		
		normcrossprod(v[0], v[1], n);
		glNode(node0);
		glNode(node1);
		glNode(node5);
		glNode(node4);

		normcrossprod(v[3], v[4], n);
		glNode(node1);
		glNode(node2);
		glNode(node6);
		glNode(node5);

		normcrossprod(v[5], v[4], n);
		glNode(node2);
		glNode(node3);
		glNode(node7);
		glNode(node6);

		normcrossprod(v[1], v[2], n);
		glNode(node3);
		glNode(node0);
		glNode(node4);
		glNode(node7);
	}	
	glEnd();

	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_LIGHTING);

	glColor3f(colorR_ ,colorG_, colorB_);
	glLineWidth(lineWidth_);
	for (int i = 0; i < size; ++i)
	{
		const Mesh::Element& element = mesh->GetConstSurfaceElement(i);
		const Mesh::Node& node0 = mesh->GetConstNode(element.GetNodeIndex(0));
		const Mesh::Node& node1 = mesh->GetConstNode(element.GetNodeIndex(1));
		const Mesh::Node& node2 = mesh->GetConstNode(element.GetNodeIndex(2));
		const Mesh::Node& node3 = mesh->GetConstNode(element.GetNodeIndex(3));
		const Mesh::Node& node4 = mesh->GetConstNode(element.GetNodeIndex(4));
		const Mesh::Node& node5 = mesh->GetConstNode(element.GetNodeIndex(5));
		const Mesh::Node& node6 = mesh->GetConstNode(element.GetNodeIndex(6));
		const Mesh::Node& node7 = mesh->GetConstNode(element.GetNodeIndex(7));

		glBegin(GL_LINE_STRIP);
			glNode(node0);
			glNode(node1);
			glNode(node2);
			glNode(node3);
			glNode(node0);

			glNode(node4);
			glNode(node5);
			glNode(node6);
			glNode(node7);
			glNode(node4);
		glEnd();

		glBegin(GL_LINES);
			glNode(node1);
			glNode(node5);
			glNode(node2);
			glNode(node6);
			glNode(node3);
			glNode(node7);
		glEnd();
	}	
}

// ����������
void MeshDrawer::drawTriPrism(const Mesh* mesh, GeometrySet* currentSelectedSet)
{
	int edgesNumber    = mesh->GetEdgesNumber();
	int verticesNumber = mesh->GetVerticesNumber();
	int elementsNumber = mesh->GetElementsNumber();

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT2);
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHT1);

	glPolygonOffset(1.0, 1.0);

	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	for(int i = 0; i < elementsNumber; i++)
	{
		//int elementFace = mesh_->get//mesh->getFaceIndexOfElement(i);
		const Mesh::Element & currentElement = mesh->GetConstElement(i);
		const Mesh::Node& Node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& Node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& Node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));
		const Mesh::Node& Node3 = mesh->GetConstNode(currentElement.GetNodeIndex(3));
		const Mesh::Node& Node4 = mesh->GetConstNode(currentElement.GetNodeIndex(4));
		const Mesh::Node& Node5 = mesh->GetConstNode(currentElement.GetNodeIndex(5));

		glBegin(GL_TRIANGLES);
			glColor3f(colorR_, colorG_, colorB_);
			//glColor3f( 0.0756800000f, 0.6142400000f, 0.0756800000f );
			//glColor3f( 0.0f, 0.0f, 0.0f );

			//������İڷ�˳��ͬ�������ɵ������εķ���Ҳ����ͬ���������ν�������ɫ������Ϊ��ɫ������Ϊ��ɫ
			//��ˣ��������θ������ڰڷ�ʱӦ����˳ʱ������ʱ�����⡣��������׼��
			glNode(Node0);
			glNode(Node1);
			glNode(Node2);
        	
			glNode(Node3);
			glNode(Node5);
			glNode(Node4);
		glEnd();

		//������İڷ�˳��ͬ�������ɵ��ı��εķ���Ҳ����ͬ�����ı��ν�������ɫ������Ϊ��ɫ������Ϊ��ɫ
		//��ˣ����ı��θ������ڰڷ�ʱӦ����˳ʱ������ʱ�����⡣��������׼��
		glBegin(GL_QUADS);
			glNode(Node1);
			glNode(Node0);
			glNode(Node3);
			glNode(Node4);

			glNode(Node0);
			glNode(Node2);
			glNode(Node5);
			glNode(Node3);

			glNode(Node2);
			glNode(Node1);
			glNode(Node4);
			glNode(Node5);
		glEnd();
	}
	glDisable(GL_COLOR_MATERIAL);


	//����
	glDisable(GL_LIGHTING);
	glColor3f(colorR_, colorG_, colorB_);
	glLineWidth(lineWidth_);
	for(int i = 0; i < elementsNumber; i++)
	{
		const Mesh::Element & currentElement = mesh->GetConstElement(i);
		const Mesh::Node& Node0 = mesh->GetConstNode(currentElement.GetNodeIndex(0));
		const Mesh::Node& Node1 = mesh->GetConstNode(currentElement.GetNodeIndex(1));
		const Mesh::Node& Node2 = mesh->GetConstNode(currentElement.GetNodeIndex(2));
		const Mesh::Node& Node3 = mesh->GetConstNode(currentElement.GetNodeIndex(3));
		const Mesh::Node& Node4 = mesh->GetConstNode(currentElement.GetNodeIndex(4));
		const Mesh::Node& Node5 = mesh->GetConstNode(currentElement.GetNodeIndex(5));

		glBegin(GL_LINE_STRIP);
			glNode(Node0);
			glNode(Node1);
			glNode(Node2);
			glNode(Node0);
			glNode(Node3);
			glNode(Node4);
			glNode(Node5);
			glNode(Node3);
		glEnd();

		glBegin(GL_LINES);
			glNode(Node2);
			glNode(Node5);
			glNode(Node1);
			glNode(Node4);
		glEnd();
	}
	return;
}

// ����������
void MeshDrawer::drawMesh(const Mesh* mesh, const MeshStatus status, GeometrySet* currentSelectedSet)
{
	// �����������޽ڵ�, ����
	if( mesh->GetNodesNumber() <= 0 ) return;

	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glMatrixMode(GL_MODELVIEW);

	glPushAttrib(GL_ENABLE_BIT);

	// ���ù��ռ����ʼ�������ɫ
	GLfloat light_ambient[] = { 0.1, 0.1, 0.1, 1.0 };
	GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat light_specular[] = { 0.1, 0.1, 0.1, 0.1 };
	GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
	
	glLightfv(GL_LIGHT2, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT2, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT2, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT2, GL_POSITION, light_position);
	glEnable(GL_LIGHT2);		
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHT1);

	GLfloat mat_ambient[] = { 0.0215f, 0.1745f, 0.0215f };
	GLfloat mat_diffuse[] = { 0.07568f, 0.61424f, 0.07568f };
	GLfloat mat_specular[]= { 0.633f, 0.633f, 0.633f };
	GLfloat mat_shiness = 0.6f;

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,  mat_ambient);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,  mat_diffuse);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, mat_shiness*128);
	
	glLineWidth(lineWidth_);
	glColor3f(colorR_, colorG_, colorB_);
	
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

	int elementType = mesh->GetElement(0).GetElementType();
	if (elementType == TRIANGLE_ELEMENT_TYPE)
	{
		drawTriangleSurface(mesh, currentSelectedSet);
	}
	else if (elementType == TETRAHEDRAL_ELEMENT_TYPE)
	{
		switch (status)
		{
		case Status_WireFrame:
			drawTetraWire(mesh, currentSelectedSet);
			break;
		case Status_NodesOnly:
			drawNodes(mesh, currentSelectedSet);
			break;
		case Status_HideLineInGray:
			{
			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT2);

			std::vector<GLuint> hitElements;
			std::vector<GLuint> hitSurfElements;
			getCutElements(mesh, hitElements, cuttingPlaneA_, cuttingPlaneB_, cuttingPlaneC_, cuttingPlaneD_);
			getShowSurfaceElements(mesh, hitSurfElements, cuttingPlaneA_, cuttingPlaneB_, cuttingPlaneC_, cuttingPlaneD_);

			drawTetraHides(mesh, hitElements,hitSurfElements, cuttingPlaneA_, cuttingPlaneB_, cuttingPlaneC_, cuttingPlaneD_);
			drawCuttingPlane();
			break;
			}
		default:
			{
			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT2);

			std::vector<GLuint> hitElements;
			std::vector<GLuint> hitSurfElements;
			getCutElements(mesh, hitElements, cuttingPlaneA_, cuttingPlaneB_, cuttingPlaneC_, cuttingPlaneD_);
			getShowSurfaceElements(mesh, hitSurfElements, cuttingPlaneA_, cuttingPlaneB_, cuttingPlaneC_, cuttingPlaneD_);
			drawTetraHides(mesh, hitElements, hitSurfElements, cuttingPlaneA_, cuttingPlaneB_, cuttingPlaneC_, cuttingPlaneD_);

			glDisable(GL_LIGHTING);
			drawCuttingPlane();
			break;
			}
		}
	}
	else if (elementType == HEAHEDRON_ELEMENT_TYPE) //�����嵥Ԫ
	{
		switch(status)
		{
		case Status_WireFrame:
			drawHexWire(mesh, currentSelectedSet);
			break;
		case Status_NodesOnly:
			drawNodes(mesh, currentSelectedSet);
			break;
		case Status_HideLineInGray:
			{
			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT2);
			std::vector<GLuint> hitElements;
			std::vector<GLuint> hitSurfElements;
			drawHexHides(mesh, hitElements, hitSurfElements, cuttingPlaneA_, cuttingPlaneB_, cuttingPlaneC_, cuttingPlaneD_);
			glDisable(GL_LIGHTING);
			break;
			}
		default:
			{
			glEnable(GL_LIGHTING);
			glEnable(GL_LIGHT2);
			std::vector<GLuint> hitElements;
			std::vector<GLuint> hitSurfElements;
			drawHexHides(mesh, hitElements, hitSurfElements, cuttingPlaneA_, cuttingPlaneB_, cuttingPlaneC_, cuttingPlaneD_);
			glDisable(GL_LIGHTING);
			break;
			}
		}
	}

	glPopAttrib();
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	glDisable(GL_LIGHT2);

	return;
}

} // namespace icesop
