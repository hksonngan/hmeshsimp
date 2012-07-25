/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 类GeometrySet:
** 包括点、边、面几何元素的集合
** 可用于指定选择的几何元素,以便对其进行操作(如渲染、修改)
**
** Author : shan @2011
**
****************************************************************************/

#ifndef GEOMETRYSET_H
#define GEOMETRYSET_H

#include <vector>
#include <algorithm>
#include <ostream>
#include <iterator>
#include <istream>
#include <iostream>

namespace icesop {

class GeometrySet
{
public:
	GeometrySet() {}

	~GeometrySet(){}

	void addVertex(unsigned vertexIndex)
	{
		for( unsigned i = 0; i < vertices.size(); i++ )
		{
			if( vertices[i] == vertexIndex )
			{
				return;
			}
		}
		vertices.push_back(vertexIndex);
	}

	void addEdge(unsigned edgeIndex)
	{
		for( unsigned i = 0; i < edges.size(); i++ )
		{
			if( edges[i] == edgeIndex )
			{
				return;
			}
		}
		edges.push_back(edgeIndex);
	}

	void addFace(unsigned faceIndex)
	{
		for( unsigned i = 0; i < faces.size(); i++ )
		{
			if( faces[i] == faceIndex )
			{
				return;
			}
		}
		faces.push_back(faceIndex);
	}

	int GetVerticesNum() const
	{
		return vertices.size();
	}

	int GetEdgesNum() const
	{
		return edges.size();
	}

	int GetFacesNum() const
	{
		return faces.size();
	}

	int GetVertexIndex(unsigned i) const
	{
		return vertices.at(i);
	}

	int GetEdgeIndex(unsigned i) const
	{
		return edges.at(i);
	}

	int GetFaceIndex(unsigned i) const
	{
		return faces.at(i);
	}

	void removeAll()
	{
		vertices.clear();
		edges.clear();
		faces.clear();
	}

	bool isEmpty() const
	{
		return( vertices.empty() && edges.empty() && faces.empty() );
	}

	bool operator==(GeometrySet& anotherSet)
	{
		if (this->edges.size() != anotherSet.edges.size()
			|| this->faces.size() != anotherSet.faces.size()
			|| this->vertices.size() != anotherSet.vertices.size())
		{
			return false;
		}
		std::sort(this->edges.begin(), this->edges.end());
		std::sort(this->faces.begin(), this->faces.end());
		std::sort(this->vertices.begin(), this->vertices.end());
		std::sort(anotherSet.edges.begin(), anotherSet.edges.end());
		std::sort(anotherSet.faces.begin(), anotherSet.faces.end());
		std::sort(anotherSet.vertices.begin(), anotherSet.vertices.end());


		if (this->edges != anotherSet.edges)
		{
			return false;
		}

		if (this->faces != anotherSet.faces)
		{
			return false;
		}

		if (this->vertices != anotherSet.vertices)
		{
			return false;
		}

		return true;
	}
    
	friend std::ostream& operator<< (std::ostream& os, const GeometrySet& gs)
	{
		os << gs.edges.size() << " " << gs.vertices.size() << " " << gs.faces.size() << "\n";
		std::copy(gs.edges.begin(), gs.edges.end(), std::ostream_iterator<unsigned>(os, "\n"));
		std::copy(gs.vertices.begin(), gs.vertices.end(), std::ostream_iterator<unsigned>(os, "\n"));
		std::copy(gs.faces.begin(), gs.faces.end(), std::ostream_iterator<unsigned>(os, "\n"));
		return os;
	}
    
	friend std::istream& operator>> (std::istream& is, GeometrySet& gs)
	{   
		size_t eNo, vNo, fNo; 
		is >> eNo >> vNo >> fNo;
		//std::cerr << "before 3 size" << std::endl;
		gs.edges.resize(eNo);
		//std::cerr << "esize" << std::endl;
		//std::cerr << eNo << " " << vNo << " " << fNo << std::endl;
		gs.vertices.resize(vNo);
		gs.faces.resize(fNo);
		//std::cerr << "finished 3 size" << std::endl;
		for (size_t i = 0; i < eNo; ++i)
		{
			is >> gs.edges[i];
		}
		//std::cerr << "gs.edges" << std::endl;
		for (size_t i = 0; i < vNo; ++i)
		{
			is >> gs.vertices[i];
		}
		
		//std::cerr << "gs.vertice";
		for (size_t i = 0; i < fNo; ++i)
		{
			is >> gs.faces[i];
		}
        return is;
	}

private:
	std::vector<unsigned> vertices;
	std::vector<unsigned> edges;
	std::vector<unsigned> faces;
};

} // namespace icesop

#endif // GEOMETRYSET_H
