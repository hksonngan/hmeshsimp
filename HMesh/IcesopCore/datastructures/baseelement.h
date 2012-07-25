/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ��BaseElement:
** ģ����,���嵥Ԫ
** ͨ������vector����node(�ڵ�)�б������ֵ�Ԫ
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
// history : ����ͱ�ֵ����ȡ ��������nodenumber�Ĳ��� ����һ�������ڵ�������һ�Զ����ȡ�� at 2008.03.05 by duguguiyu
//           ȥ���ͱ���ȡ���� �ø��ϵ�typeֵ����type��һЩ��ֵ��Ϣ at 2008.05.08 by duguguiyu
//           ��Ϊ����ʱȷ����Ԫ���� by duguguiyu at 2008-05-23
//
// history : Ϊ��Ԫ���Ӽ������������Ͷ����������� by LiShilei at 2009-12-04
////////////////////////////////////////////////////////////////

#ifndef BASEELEMENT_H
#define BASEELEMENT_H

#include <vector>
#include <assert.h>

namespace icesop {

/** @addtogroup Kernel
 *  @{ */
const unsigned VERTEX_ELEMENT_TYPE      = 101;  ///< �����͵�Ԫ
const unsigned LINE_ELEMENT_TYPE        = 202;  ///< �����͵�Ԫ
const unsigned TRIANGLE_ELEMENT_TYPE    = 303;  ///< �����浥Ԫ
const unsigned TETRAHEDRAL_ELEMENT_TYPE = 304;  ///< �����嵥Ԫ
const unsigned RECTANGLE_ELEMENT_TYPE   = 404;  ///< �ı��ε�Ԫ
const unsigned HEAHEDRON_ELEMENT_TYPE   = 408;  ///< �����嵥Ԫ
/** @} */

/**  ��ʾ�����е�һ����Ԫ.
 *   �����ͱ�ʾ�����ݶ����£�
 *   @li �Ͱ�λ��ʾ�ڵ�����
 *   @li �߰�λ��ʾ����ڵ�����
 *   ���磬������������304��ʾ��
 *   @todo ������ԵĲ����꾡������̬����Ӧ�ö���һ��
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

	// ��õ�Ԫһ����Ľڵ�����.
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
	// �������г�ȡ�ڵ�����.
	static unsigned GetNodeNumber(unsigned elementType)
	{
		return elementType % 100;
	}

	// �������г�ȡ��ڵ�����.
	static unsigned GetSurfaceNodeNumber(unsigned elementType)
	{
		return (elementType / 100) % 100;
	}

	// �ǲ���һ���浥Ԫ.
	static bool IsSurfaceElement(unsigned elementType)
	{
		return GetNodeNumber(elementType) == GetSurfaceNodeNumber(elementType);
	}

	// ��ȡ���浥Ԫ�ĵ�Ԫ����.
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
