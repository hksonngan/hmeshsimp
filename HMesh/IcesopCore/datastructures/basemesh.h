/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ��BaseMesh:
** ģ����,��������
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
// history : ��Ϊ����ʱȷ��ά�Ⱥ����� by duguguiyu at 2008-05-23
//           ������һЩԤ����ӿ� ������const &��ԭ�еĴ�ֵ���иĽ� ��Ϊ����������ѧ֮�겻�����в��е����� by duguguiyu at 2008-05-29
//           ��ӶԻ�������֧�� by duguguiyu at 2008-06-18
// history : ����˼������ԺͶ������Ե�֧�� by LiShilei at 2009-12-06
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

/**  ����ڵ��һ���Ƚ���.
 *   ����һ�����ֵ�������ָ�룬�ɽ��е�ֵ����С�ȱȽϡ�
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

	/**  ����һЩSTL���㷨��Ĭ����С�ڱȽ�.
	 *
	 */
	bool operator() (const unsigned leftNodeIndex, const unsigned rightNodeIndex) const
	{
		return LessThen(leftNodeIndex, rightNodeIndex);
	}

	/**  С�ڱȽ�.
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

	/**  ��ֵ�Ƚ�.
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

/**  ��ʾ����.
 *   ����Ԥ������֧�ֵ�ά�Ⱥ��������ͣ�ͬʱ��֧�����ļ���ȡ��̬ȷ����
 *   �߱�һЩ����ĺ��ڴ����ܣ����磺ȥ�أ�ȥ����֮��ġ�
 *   @todo Clear������Ҫ˼��һ�£�Ӧ���������ڲ�������ֵ�������Ǽ���
 *   @author dugugiyu
 *   @date 2007-12-29
 *   @ingroup Kernel
 */
template< typename T >
class BaseMesh
{
public:
	typedef BaseNode<T> Node;  ///< �ڵ�����
	typedef BaseElement Element; ///<  ��Ԫ����
	typedef BaseMaterial<T> Material; ///< ��������
	typedef BaseCollection<Element> Elements; ///< ��Ԫ��������
	typedef BaseCollection<Node> Nodes; ///< �ڵ㼯������
	typedef BaseCollection<Material> Materials; ///< ���ϼ�������
	typedef std::vector<unsigned> Edge; ///< ������ һ���ߣ���Ӧ��ı�
	typedef BaseCollection<Edge> Edges; ///< �߼�������
	typedef BaseCollection<unsigned> Vertices; ///<���㼯��

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
	/** �������캯��.
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

	/** ��ֵ����.
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
	/**  У���Ƿ��ǺϷ��Ľڵ�.
	 *   ��Ҫ����ά���ϵļ�飬���߼��ϵĺϷ��Բ�����顣
	 *   ��Ҫ�ֶ����ã�������AddNode��ֱ��assertƮ��������exception
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
	/**  ���ؽڵ㼯�ϵ�ָ��.
	 *   ���Կ�����������Ż���Ҫ���ͱ��const��Ҫ���͸ɴ��ɴ�ֵ��
	 */
	Nodes * GetNodes() const
	{
		return _nodes;
	}


	/**  �����֧�ֵ�����Ԫ����.
	 *
	 */
	void AddSupportElementType(unsigned elementType)
	{
		this->_elementIndexs.AddKey(elementType);
	}

	/**  �����֧������Ԫ���͵���Ŀ.
	 *
	 */
	unsigned GetSupportElementTypeNumber() const
	{
		return this->_elementIndexs.GetKeysNumber();
	}

	/**  �Ƿ��ǵ���Ԫ������.
	 *
	 */
	bool IsSingleElementMesh() const
	{
		return this->GetSupportElementTypeNumber() == 1;
	}

	/**  ������֧�ֵ�����Ԫ���ͣ������ص�һ����֧�ֵĵ�Ԫ���ͣ������ڵ���Ԫ�����.
	 *
	 */
	unsigned GetSupportElementType() const
	{
		assert(GetSupportElementTypeNumber() > 0);

		return this->_elementIndexs.GetKeyBegin()->first;
	}

	/**  ������֧�ֵ�����Ԫ���ͣ��������е�����.
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

	/**  ����Ƿ����Ѿ�֧�ֵ���������.
	 *
	 */
	bool IsLegalElement(const Element & element) const
	{
		return this->_elementIndexs.HaveTheKey(element.GetElementType());
	}

	/**  �������Ԫ��Ԫ.
	 *   �����ﲻ�����κι�������Ԫ��Ԫ֧���Եļ�飬�������Ч�ʺͷ����ȡ������
	 *   ��Ҫά�����߼��ĵط���Ӧ���ȵ���IsLegalElementȻ���ٵ��øú�����
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


	/** ���������ı���ڵ㵥Ԫ
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

	/**  �����ɢ��.
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


	/**  ����������ݣ�����Ԥ��ֵ.
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

	/**  ����������ݣ�����ά�Ⱥ�֧�����������.
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
	 * ��ȡBaseMesh����İ�Χ��
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
	/**  ��ָ���ĸ������ȥ���ظ��Ľڵ�.
	 *   �����ǶԽڵ���п��ţ�Ȼ��ȥ�أ�����¼�仯����Ϣ��
	 *   �����ݼ�¼�Ľű�仯��Ϣ��element���е�����
	 */
	void UniqueNodes(const double tolerance = 0.000001)
	{
		//  ����ȥ��
		std::vector<unsigned> indexMap(GetNodesNumber());
		UniqueNodesCollection(tolerance, indexMap);

		//  ���µ�Ԫ
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

	/**  �Ƴ���Element������δ���ֵĽڵ㣬�����Ƿ�����˵�����newNodeIndexsΪ�������������¾ɽű��Ӧ.
	 *   �ú���У��ڵ��Ƿ���Element�г��֣�ͬʱУ��Element���Ƿ���ڲ��������еĽڵ㡣
	 *   ��ˣ���Ҫ�����г�ʼ��������ɺ���иù�����
	 */
	bool RemoveUnusedNodes(std::vector<unsigned>& newNodeIndexs)
	{
		if(this->GetNodesNumber() == 0)
			return false;

		// *** ͳ�Ƴ��ֵĽڵ� ***
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

		// *** �����µĽڵ�ű� ***
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

		if(!needAdjust) // �������Ҫ�ı� ����
			return false;

		// *** �����µĽڵ���***
		Nodes * newNodes = new Nodes();
		for(unsigned nodeIndex = 0; nodeIndex < this->GetNodesNumber(); ++nodeIndex)
		{
			if(visited[nodeIndex])
				newNodes->AddValue(GetNode(nodeIndex));
		}
		delete _nodes;
		_nodes = newNodes;

		// *** ����Element�ű� ***
		RefreshElementNodes(newNodeIndexs);

		return true;
	}

private:
	/**  ...
	 *
	 */
	void UniqueNodesCollection(double const tolerance, std::vector<unsigned>& indexMap)
	{
		//  ������������
		std::vector<unsigned> nodeIndexs(GetNodesNumber());
		for(unsigned i = 0; i < nodeIndexs.size(); ++i)
			nodeIndexs[i] = i;

		MeshNodeComparator<T> comparator(this, tolerance);

		//  ����
		std::sort(nodeIndexs.begin(), nodeIndexs.end(), comparator);

		Nodes * newNodes = new Nodes();
		unsigned lastIndex = 0;
		unsigned lastUniqueIndex = newNodes->AddValue(this->GetNode(nodeIndexs[lastIndex]));
		indexMap[nodeIndexs[lastIndex]] = lastUniqueIndex;

		const unsigned ununiqueNodesNumber = GetNodesNumber();
		for(unsigned index = 1; index < ununiqueNodesNumber; ++index) // ��1��ʼ����
		{
			unsigned currentIndex = nodeIndexs[index];
			if( !comparator.IsEquals(lastIndex, currentIndex) ) // �������ǰһ��ֵ�������뵽mesh��
			{
				lastIndex = currentIndex;
				lastUniqueIndex = newNodes->AddValue(this->GetNode(currentIndex));;
			}

			indexMap[currentIndex] = lastUniqueIndex;
		}

		delete this->_nodes;
		this->_nodes = newNodes;
	}
	/**  ����һ���¾ɽڵ���Ҫӳ��������½ڵ����.
	 *
	 */
	void RefreshElementNodes(std::vector<unsigned> &newNodeIndexs)
	{
		for(unsigned elementIndex = 0; elementIndex < GetElementsNumber(); ++elementIndex)
		{
			/*
			 *  Mark ��Ч�Ĵ���ģʽ ��Ҫ��ӶԻ������õĽӿ�
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
	Elements * _elements;  //  ��Ԫ����
	BaseIndexs<unsigned> _elementIndexs;  //  ���ڵ�Ԫ���͵ĵ�������
	std::vector<unsigned> _faceIndexOfElements;// ��¼��Ԫ����ԭʵ�����Ƭ�ı��
	Edges* _edges;
	Vertices* _vertices;
	Elements *_surfaceElements;   //��Ԫ����Ϊ������ʱ��¼������ı��棬��Ԫ����Ϊ������Ϊ��
	Materials * _materials;
};

} // namespace icesop


#endif // BASEMESH_H
