/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 类BaseMesh:
** 模板类,定义网格
**
** Author : shan @2011
**
****************************************************************************/

////////////////////////////////////////////////////////////////
//
// author: dugugiyu
// date  : 2007-12-29
//
// description:
//
// history : 改为运行时确定维度和其他 by duguguiyu at 2008-05-23
//           调整了一些预处理接口 并且用const &对原有的传值进行改进 因为考虑在我有学之年不会再有并行的问题 by duguguiyu at 2008-05-29
//           添加对混合网格的支持 by duguguiyu at 2008-06-18
// history : 添加了几何属性和额外属性的支持 by LiShilei at 2009-12-06
////////////////////////////////////////////////////////////////

#ifndef BASEMESH_H
#define BASEMESH_H

#include "basenode.h"
#include "baseelement.h"
#include "basematerial.h"
#include "baseindexs.h"

#include <vector>
#include <algorithm>
#include <functional>
#include <math.h>
#include <assert.h>

namespace icesop {

template<typename T>
class BaseMesh;

/**  网格节点的一个比较器.
 *   传入一个误差值和网格的指针，可进行等值、大小等比较。
 *   @author dugugiyu
 *   @date 2008-05-29
 *   @ingroup Kernel
 */
template<typename T>
class MeshNodeComparator
{
public:
	/**  ...
	 *
	 */
	MeshNodeComparator(const BaseMesh<T> * mesh, const double tolerance) : _mesh(mesh), _tolerance(tolerance)
	{
	}

	/**  用与一些STL的算法，默认是小于比较.
	 *
	 */
	bool operator() (const unsigned leftNodeIndex, const unsigned rightNodeIndex) const
	{
		return LessThen(leftNodeIndex, rightNodeIndex);
	}

	/**  小于比较.
	 *
	 */
	bool LessThen(const unsigned leftNodeIndex, const unsigned rightNodeIndex) const
	{
		const typename BaseMesh<T>::Node leftNode(_mesh->GetNode(leftNodeIndex));
		const typename BaseMesh<T>::Node rightNode(_mesh->GetNode(rightNodeIndex));

		unsigned dimension = leftNode.GetDimension();
		for(unsigned i = 0; i < dimension; ++i)
		{
			double diffX = leftNode.GetLocationValue(i)  - rightNode.GetLocationValue(i);
			if( diffX < -_tolerance )
				return true;
			if( diffX >  _tolerance )
				return false;
		}

		return false;
	}

	/**  等值比较.
	 *
	 */
	bool IsEquals(const unsigned leftNodeIndex, const unsigned rightNodeIndex) const
	{
		const typename BaseMesh<T>::Node leftNode(_mesh->GetNode(leftNodeIndex));
		const typename BaseMesh<T>::Node rightNode(_mesh->GetNode(rightNodeIndex));

		for(unsigned i = 0; i < leftNode.GetDimension(); ++i)
		{
			if(abs(leftNode.GetLocationValue(i)  - rightNode.GetLocationValue(i)) > _tolerance)
				return false;
		}

		return true;
	}

private:
	const BaseMesh<T> * _mesh;
	const double _tolerance;
};

/**  表示网格.
 *   可以预先设置支持的维度和网格类型，同时，支持由文件读取动态确定。
 *   具备一些网格的后期处理功能，比如：去重，去冗余之类的。
 *   @todo Clear方法需要思考一下，应该是消除内部索引的值，而不是键。
 *   @author dugugiyu
 *   @date 2007-12-29
 *   @ingroup Kernel
 */
template< typename T >
class BaseMesh
{
public:
	typedef BaseNode<T> Node;  ///< 节点类型
	typedef BaseElement Element; ///<  单元类型
	typedef BaseMaterial<T> Material; ///< 材料类型
	typedef BaseCollection<Element> Elements; ///< 单元集合类型
	typedef BaseCollection<Node> Nodes; ///< 节点集合类型
	typedef BaseCollection<Material> Materials; ///< 材料集合类型
	typedef std::vector<unsigned> Edge; ///< 边类型 一条边，对应体的边
	typedef BaseCollection<Edge> Edges; ///< 边集合类型
	typedef BaseCollection<unsigned> Vertices; ///<顶点集合

public:
	/** ...
	 *
	 */
	BaseMesh() : _initDimension(0)
	{
		this->_nodes = new Nodes();
		this->_elements = new Elements();
		this->_materials = new Materials();
		this->_edges = new Edges();
		this->_vertices = new Vertices();
		this->_surfaceElements = new Elements();
	}
	/** ...
	 *
	 */
	~BaseMesh()
	{
		delete this->_nodes;
		delete this->_elements;
		delete this->_materials;
		delete this->_vertices;
		delete this->_edges;
	}

public:
	/** 拷贝构造函数.
	 *
	 */
	BaseMesh(const BaseMesh<T> & rhs)
	{
		this->_nodes = new Nodes(*rhs._nodes);
		this->_elements = new Elements(*rhs._elements);
		this->_materials = new Materials(*rhs._materials);
		this->_edges = new Edges(*rhs._edges);
		this->_vertices = new Vertices(*rhs._vertices);
		this->_surfaceElements = new Elements(*rhs._surfaceElements);
		this->_initDimension = rhs._initDimension;
		this->_elementIndexs = rhs._elementIndexs;
		this->_faceIndexOfElements = rhs._faceIndexOfElements;
	}

	/** 赋值函数.
	 *
	 */
	BaseMesh<T> & operator = (const BaseMesh<T> & rhs)
	{
		this->_nodes = new Nodes(*rhs._nodes);
		this->_elements = new Elements(*rhs._elements);
		this->_materials = new Materials(*rhs._materials);
		this->_edges = new Edges(*rhs._edges);
		this->_vertices = new Vertices(*rhs._vertices);
		this->_surfaceElements = new Elements(*rhs._surfaceElements);
		this->_initDimension = rhs._initDimension;
		this->_elementIndexs = rhs._elementIndexs;
		this->_faceIndexOfElements = rhs._faceIndexOfElements;

		return *this;
	}

public:
	/**  校验是否是合法的节点.
	 *   主要是做维度上的检查，对逻辑上的合法性不做检查。
	 *   需要手动调用，否则在AddNode中直接assert飘出而不是exception
	 */
	bool IsLegalNode(const Node & node)
	{
		return GetDimension() == 0 || node.GetDimension() == GetDimension();
	}

	/** ..
	 *
	 */
	unsigned AddNode(const Node & node)
	{
		assert(IsLegalNode(node));

		return _nodes->AddValue(node);
	}
	/** ..
	 *
	 */
	void SetNode(const Node & node, unsigned index)
	{
		assert(index < GetNodesNumber());
		assert(IsLegalNode(node));

		_nodes->SetValue(node, index);
	}
	/** ..
	 *
	 */
	Node GetNode(unsigned index) const
	{
		assert(index < GetNodesNumber());

		return _nodes->GetValue(index);
	}
	const Node & GetConstNode(unsigned index) const
	{
		return _nodes->GetConstValue(index);
	}
	/** ..
	 *
	 */
	unsigned GetNodesNumber() const
	{
		return _nodes->Size();
	}
	/**  返回节点集合的指针.
	 *   可以考虑两方面的优化，要不就变成const，要不就干脆变成传值。
	 */
	Nodes * GetNodes() const
	{
		return _nodes;
	}


	/**  添加所支持的有限元类型.
	 *
	 */
	void AddSupportElementType(unsigned elementType)
	{
		this->_elementIndexs.AddKey(elementType);
	}

	/**  获得所支持有限元类型的数目.
	 *
	 */
	unsigned GetSupportElementTypeNumber() const
	{
		return this->_elementIndexs.GetKeysNumber();
	}

	/**  是否是单单元的网格.
	 *
	 */
	bool IsSingleElementMesh() const
	{
		return this->GetSupportElementTypeNumber() == 1;
	}

	/**  返回所支持的有限元类型，仅返回第一个所支持的单元类型，适用于单单元的情况.
	 *
	 */
	unsigned GetSupportElementType() const
	{
		assert(GetSupportElementTypeNumber() > 0);

		return this->_elementIndexs.GetKeyBegin()->first;
	}

	/**  返回所支持的有限元类型，包含所有的类型.
	 *
	 */
	std::vector<unsigned> GetSupportElementTypes() const
	{
		std::vector<unsigned> supportElementTypes;

		BaseIndexs<unsigned>::Keys::const_iterator end = this->_elementIndexs.GetKeyEnd();

		for(BaseIndexs<unsigned>::Keys::const_iterator iter = this->_elementIndexs.GetKeyBegin();
			iter != end;
			++iter)
		{
			supportElementTypes.push_back(iter->first);
		}

		return supportElementTypes;
	}

	/** ...
	 *
	 */
	void ClearSupportElementTypes()
	{
		this->_elementIndexs.Clear();
	}

	/**  检查是否是已经支持的网格类型.
	 *
	 */
	bool IsLegalElement(const Element & element) const
	{
		return this->_elementIndexs.HaveTheKey(element.GetElementType());
	}

	/**  添加有限元单元.
	 *   在这里不会做任何关于有限元单元支持性的检查，用以提高效率和方便读取操作。
	 *   需要维护该逻辑的地方，应该先调用IsLegalElement然后再调用该函数。
	 */
	unsigned AddElement(const Element & element)
	{
		unsigned index = _elements->AddValue(element);
		this->_elementIndexs.AddIndex(element.GetElementType(), index);

		return index;
	}

	/** ...
	 *
	 */
	void SetElement(Element element, unsigned index)
	{
		assert(index < GetElementsNumber());
		assert(IsLegalElement(element));

		_elements->SetValue(element, index);
	}

	/** ...
	 *
	 */
	Element GetElement(unsigned index) const
	{
		//assert(index < GetElementsNumber());
		return _elements->GetValue(index);
	}

	const Element& GetConstElement(unsigned index) const
	{
		return _elements->GetConstValue(index);
	}

	/** ...
	 *
	 */
	unsigned GetElementsNumber() const
	{
		return _elements->Size();
	}


	/** 添加体网格的表面节点单元
	*
	*/
	unsigned AddSurfaceElement(const Element & element)
	{
		unsigned index = _surfaceElements->AddValue(element);
		this->_elementIndexs.AddIndex(element.GetElementType(), index);

		return index;
	}

	/** ...
	 *
	 */
	void SetSurfaceElement(Element element, unsigned index)
	{
		assert(index < GetSurfaceElementsNumber());
		assert(IsLegalElement(element));

		_surfaceElements->SetValue(element, index);
	}

	/** ...
	 *
	 */
	Element GetSurfaceElement(unsigned index) const
	{
		//assert(index < GetElementsNumber());
		return _surfaceElements->GetValue(index);
	}

	const Element& GetConstSurfaceElement(unsigned index) const
	{
		return _surfaceElements->GetConstValue(index);
	}

	/** ...
	 *
	 */
	unsigned GetSurfaceElementsNumber() const
	{
		return _surfaceElements->Size();
	}

	/**  添加离散边.
	*
	*/
	unsigned AddEdge(const Edge & edge)
	{
		unsigned index = _edges->AddValue(edge);
		return index;
	}

	/** ...
	*
	*/
	void SetEdge(Edge edge, unsigned index)
	{
		_edges->SetValue(edge, index);
	}

	/** ...
	*
	*/
	Edge GetEdge(unsigned index) const
	{
		return _edges->GetValue(index);
	}

	const Edge& GetConstEdge(unsigned index) const
	{
		return _edges->GetConstValue(index);
	}

	/** ...
	*
	*/
	unsigned GetEdgesNumber() const
	{
		return _edges->Size();
	}

	unsigned AddVertex(const unsigned node_index)
	{
		return _vertices->AddValue(node_index);
	}

	unsigned GetVerticesNumber() const
	{
		return _vertices->Size();
	}

	unsigned GetVertex(unsigned i) const
	{
		return _vertices->GetValue(i);
	}

	const unsigned GetConstVertex(unsigned i) const
	{
		return _vertices->GetValue(i);
	}


	void setFaceIndexOfElement(unsigned faceIndex, unsigned elementIndex)
	{
		_faceIndexOfElements[elementIndex] = faceIndex;
	}

	unsigned getFaceIndexOfElement( unsigned elementIndex ) const
	{
		return _faceIndexOfElements[elementIndex];
	}

	unsigned pushFaceIndexOfElement(unsigned faceIndex)
	{
		_faceIndexOfElements.push_back(faceIndex);
		return _faceIndexOfElements.size();
	}

	/** ...
	 *
	 */
	Materials * GetMaterials() const
	{
		return this->_materials;
	}

	/** ...
	 *
	 */
	unsigned AddMaterial(const Material& material)
	{
		return _materials->AddValue(material);
	}

	/** ...
	 *
	 */
	Material GetMaterial(unsigned index) const
	{
		assert(index < _materials->Size());
		return this->_materials->GetValue(index);
	}

	/** ...
	 *
	 */
	unsigned GetMaterialNumber() const
	{
		return this->_materials->Size();
	}


	/**  清除所有内容，包括预设值.
	 *
	 */
	void Clear()
	{
		this->_nodes->Clear();
		this->_elements->Clear();
		this->_materials->Clear();

		_elementIndexs.Clear();
		this->_initDimension = 0;
	}

	/**  清除所有数据，保留维度和支持网格等内容.
	 *
	 */
	void ClearData()
	{
		this->_nodes->Clear();
		this->_elements->Clear();
		this->_materials->Clear();

		this->_elementIndexs.ClearData();
	}


	/** ...
	 *
	 */
	void SetDimension(unsigned dimension)
	{
		assert(this->GetNodesNumber() == 0);

		this->_initDimension = dimension;
	}
	/** ...
	 *
	 */
	unsigned GetDimension() const
	{
		if(this->_initDimension > 0)
			return this->_initDimension;

		if(this->GetNodesNumber() > 0)
			return _nodes->GetValue(0).GetDimension();

		return this->_initDimension;
	}

	/**
	 * 获取BaseMesh网格的包围盒
	 */
	void GetBounds(Node& minNode, Node& maxNode)
	{
		unsigned nodeNum = this->GetNodesNumber();

		if(nodeNum > 0)
		{
			Node firstNode = this->GetNode(0);
    		for(unsigned i = 0; i < firstNode.GetDimension(); ++i)
			{
				minNode.SetLocationValue(firstNode.GetLocationValue(i), i);
				maxNode.SetLocationValue(firstNode.GetLocationValue(i), i);
			}
    	
			for(unsigned j = 1; j < nodeNum; ++j)
			{
				Node boundsNode = this->GetNode(j);
				for(unsigned i = 0; i < firstNode.GetDimension(); ++i)
				{
					if(boundsNode.GetLocationValue(i) < minNode.GetLocationValue(i))
						minNode.SetLocationValue(boundsNode.GetLocationValue(i), i);
					if(boundsNode.GetLocationValue(i) > maxNode.GetLocationValue(i))
						maxNode.SetLocationValue(boundsNode.GetLocationValue(i), i);
				}
			}
		}
	}


public:
	/**  按指定的各边误差去除重复的节点.
	 *   方法是对节点进行快排，然后去重，并记录变化的信息。
	 *   最后根据记录的脚标变化信息对element进行调整。
	 */
	void UniqueNodes(const double tolerance = 0.000001)
	{
		//  排序并去重
		std::vector<unsigned> indexMap(GetNodesNumber());
		UniqueNodesCollection(tolerance, indexMap);

		//  更新单元
		RefreshElementNodes(indexMap);
	}

	/** ..
	 *
	 */
	bool RemoveUnusedNodes()
	{
		std::vector<unsigned> newNodeIndexs(this->GetNodesNumber(), 0);
		return RemoveUnusedNodes(newNodeIndexs);
	}

	/**  移除在Element集合中未出现的节点，返回是否进行了调整，newNodeIndexs为输出参数，输出新旧脚标对应.
	 *   该函数校验节点是否在Element中出现，同时校验Element中是否存在不在序列中的节点。
	 *   因此，需要在所有初始化工作完成后进行该工作。
	 */
	bool RemoveUnusedNodes(std::vector<unsigned>& newNodeIndexs)
	{
		if(this->GetNodesNumber() == 0)
			return false;

		// *** 统计出现的节点 ***
		std::vector<bool> visited(this->GetNodesNumber(),false);
		for(unsigned elementIndex = 0; elementIndex < GetElementsNumber(); ++elementIndex)
		{
			Element currentElement = GetElement(elementIndex);

			for(unsigned nodePos = 0; nodePos < currentElement.GetNodeNumber(); ++nodePos)
			{
				assert(currentElement.GetNodeIndex(nodePos) < this->GetNodesNumber());
				visited[currentElement.GetNodeIndex(nodePos)] = true;
			}
		}

		// *** 计算新的节点脚标 ***
		assert(static_cast<unsigned>(newNodeIndexs.size()) == this->GetNodesNumber());
		unsigned baseIndex = 0;
		bool needAdjust = false;
		for(unsigned nodeIndex = 0; nodeIndex < this->GetNodesNumber(); ++nodeIndex)
		{
			if(visited[nodeIndex])
			{
				newNodeIndexs[nodeIndex] = nodeIndex - baseIndex;
			}
			else
			{
				++baseIndex;
				needAdjust = true;
			}
		}

		if(!needAdjust) // 如果不需要改变 返回
			return false;

		// *** 生成新的节点结合***
		Nodes * newNodes = new Nodes();
		for(unsigned nodeIndex = 0; nodeIndex < this->GetNodesNumber(); ++nodeIndex)
		{
			if(visited[nodeIndex])
				newNodes->AddValue(GetNode(nodeIndex));
		}
		delete _nodes;
		_nodes = newNodes;

		// *** 更新Element脚标 ***
		RefreshElementNodes(newNodeIndexs);

		return true;
	}

private:
	/**  ...
	 *
	 */
	void UniqueNodesCollection(double const tolerance, std::vector<unsigned>& indexMap)
	{
		//  建立索引序列
		std::vector<unsigned> nodeIndexs(GetNodesNumber());
		for(unsigned i = 0; i < nodeIndexs.size(); ++i)
			nodeIndexs[i] = i;

		MeshNodeComparator<T> comparator(this, tolerance);

		//  排序
		std::sort(nodeIndexs.begin(), nodeIndexs.end(), comparator);

		Nodes * newNodes = new Nodes();
		unsigned lastIndex = 0;
		unsigned lastUniqueIndex = newNodes->AddValue(this->GetNode(nodeIndexs[lastIndex]));
		indexMap[nodeIndexs[lastIndex]] = lastUniqueIndex;

		const unsigned ununiqueNodesNumber = GetNodesNumber();
		for(unsigned index = 1; index < ununiqueNodesNumber; ++index) // 从1开始遍历
		{
			unsigned currentIndex = nodeIndexs[index];
			if( !comparator.IsEquals(lastIndex, currentIndex) ) // 如果不与前一个值相等则插入到mesh中
			{
				lastIndex = currentIndex;
				lastUniqueIndex = newNodes->AddValue(this->GetNode(currentIndex));;
			}

			indexMap[currentIndex] = lastUniqueIndex;
		}

		delete this->_nodes;
		this->_nodes = newNodes;
	}
	/**  根据一个新旧节点需要映射表来更新节点序号.
	 *
	 */
	void RefreshElementNodes(std::vector<unsigned> &newNodeIndexs)
	{
		for(unsigned elementIndex = 0; elementIndex < GetElementsNumber(); ++elementIndex)
		{
			/*
			 *  Mark 低效的处理模式 需要添加对基于引用的接口
			 */
			Element currentElement = GetElement(elementIndex);

			for(unsigned nodePos = 0; nodePos < currentElement.GetNodeNumber(); ++nodePos)
			{
				currentElement.SetNodeIndex(newNodeIndexs[currentElement.GetNodeIndex(nodePos)], nodePos);
			}

			SetElement(currentElement, elementIndex);
		}
	}

	friend std::ostream& operator <<(std::ostream &os, const BaseMesh<T> & mesh)
	{   
		os << "dimension is " << mesh._initDimension << std::endl;
		os << "nodes' number is" << mesh._nodes->Size() << std::endl;
		os << "elements' number is" << mesh._elements->Size() << std::endl;
		return os;
	}



private:
	unsigned _initDimension;
	Nodes * _nodes;
	Elements * _elements;  //  单元集合
	BaseIndexs<unsigned> _elementIndexs;  //  基于单元类型的倒排索引
	std::vector<unsigned> _faceIndexOfElements;// 记录单元所在原实体大面片的编号
	Edges* _edges;
	Vertices* _vertices;
	Elements *_surfaceElements;   //单元类型为体网格时记录体网格的表面，单元类型为面网格为空
	Materials * _materials;
};

} // namespace icesop


#endif // BASEMESH_H
