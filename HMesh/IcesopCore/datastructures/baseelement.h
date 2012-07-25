/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 类BaseElement:
** 模板类,定义单元
** 通过定义vector保存node(节点)列表来表现单元
**
** Author : shan @2011
**
****************************************************************************/

////////////////////////////////////////////////////////////////
//
// author: dugugiyu
// date  : 2007-12-27
//
// description:
//
// history : 添加型别到值得萃取 用以消除nodenumber的参数 构成一个编译期的整形数一对多的萃取表 at 2008.03.05 by duguguiyu
//           去除型别萃取机制 用复合的type值表征type的一些数值信息 at 2008.05.08 by duguguiyu
//           改为运行时确定单元类型 by duguguiyu at 2008-05-23
//
// history : 为单元增加几何属性索引和额外属性索引 by LiShilei at 2009-12-04
////////////////////////////////////////////////////////////////

#ifndef BASEELEMENT_H
#define BASEELEMENT_H

#include <vector>
#include <assert.h>

namespace icesop {

/** @addtogroup Kernel
 *  @{ */
const unsigned VERTEX_ELEMENT_TYPE      = 101;  ///< 点类型单元
const unsigned LINE_ELEMENT_TYPE        = 202;  ///< 线类型单元
const unsigned TRIANGLE_ELEMENT_TYPE    = 303;  ///< 三角面单元
const unsigned TETRAHEDRAL_ELEMENT_TYPE = 304;  ///< 四面体单元
const unsigned RECTANGLE_ELEMENT_TYPE   = 404;  ///< 四边形单元
const unsigned HEAHEDRON_ELEMENT_TYPE   = 408;  ///< 六面体单元
/** @} */

/**  表示网格中的一个单元.
 *   其类型表示规则暂定如下：
 *   @li 低百位表示节点数量
 *   @li 高百位表示表面节点数量
 *   比如，四面体网格，用304表示。
 *   @todo 该类测试的不够详尽，各静态函数应该都测一下
 *   @author dugugiyu
 *   @date 2008-06-05
 *   @ingroup Kernel
 */
class BaseElement
{
public:
	BaseElement(unsigned elementType)
	: _elementType(elementType), _materialIndex(0), 
	  _geomiAttriIndex(0), _additionalAttriIndex(0), _nodeIndexList(GetNodeNumber(elementType))
	{
	}

	BaseElement(unsigned elementType, const unsigned * indexs)
		: _elementType(elementType), _materialIndex(0),
		_geomiAttriIndex(0), _additionalAttriIndex(0), _nodeIndexList(indexs, indexs + GetNodeNumber(elementType))
	{
	}

	unsigned GetNodeIndex(unsigned nodePos) const
	{
		assert(nodePos < this->GetNodeNumber());
		return this->_nodeIndexList[nodePos];
	}

	void SetNodeIndex(unsigned index, unsigned nodePos)
	{
		assert(nodePos < this->GetNodeNumber());
		this->_nodeIndexList[nodePos] = index;
	}

	unsigned GetElementType() const
	{
		return this->_elementType;
	}

	// 获得单元一个面的节点数量.
	unsigned GetSurfaceNodeNumber() const
	{
		return GetSurfaceNodeNumber(this->_elementType);
	}

	unsigned GetNodeNumber() const
	{
		return GetNodeNumber(this->_elementType);
	}

	unsigned GetMaterialIndex() const
	{
		return this->_materialIndex;
	}

	void SetMaterialIndex(unsigned index)
	{
		this->_materialIndex = index;
	}

	unsigned GetGeomiAttriIndex() const
	{
		return this->_geomiAttriIndex;
	}

	void SetGeomiAttriIndex(unsigned index)
	{
		this->_geomiAttriIndex = index;
	}
				
	unsigned GetAdditionalAttriIndex() const
	{
		return this->_additionalAttriIndex;
	}

	void SetAdditionalAttriIndex(unsigned index)
	{
		this->_additionalAttriIndex = index;
	}

public:
	// 从类型中抽取节点数量.
	static unsigned GetNodeNumber(unsigned elementType)
	{
		return elementType % 100;
	}

	// 从类型中抽取面节点数量.
	static unsigned GetSurfaceNodeNumber(unsigned elementType)
	{
		return (elementType / 100) % 100;
	}

	// 是不是一个面单元.
	static bool IsSurfaceElement(unsigned elementType)
	{
		return GetNodeNumber(elementType) == GetSurfaceNodeNumber(elementType);
	}

	// 提取其面单元的单元类型.
	static unsigned GetSurfaceElementType(unsigned elementType)
	{
		unsigned surfaceNodeNumber = GetSurfaceNodeNumber(elementType);
		return surfaceNodeNumber * unsigned(101);
	}

private:
	unsigned _elementType;
	std::vector<unsigned> _nodeIndexList;
	unsigned _materialIndex;
	unsigned _geomiAttriIndex;
	unsigned _additionalAttriIndex;
};

} // namespace icesop

#endif // BASEELEMENT_H
