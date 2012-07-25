/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ��BaseNode:
** ģ����,����ڵ�
** ͨ������vector����dimension(ά��)������
** ���ԶԸ�ά�����ݽ��в���
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
// history : �Ƴ�����Լ����صĽӿ� by duguguiyu at 2008-01-18
//           �Ƴ�����Լ������ر��� by duguguiyu at 2008-02-25
//           ��Ϊ����ʱȷ��ά�� by duguguiyu at 2008-05-23
//           ��һЩint���͵�����Ϊunsigned���͵� by duguguiyu at 2008-05-24
//       
////////////////////////////////////////////////////////////////

#ifndef BASENODE_H
#define BASENODE_H

#include <vector>
#include <assert.h>

namespace icesop {

/**  ��ʾ�����е�һ���ڵ�.
 *   @todo ����Ӳ���
 *   @author dugugiyu
 *   @date 2007-12-27
 *   @ingroup Kernel
 */
template< typename T >
class BaseNode
{
public:
	/**  ...
	 * 
	 */
	BaseNode(unsigned dimension) : _values(dimension)
	{
		assert(GetDimension() > 0);
	}

	/**  ...
	 * 
	 */
	BaseNode(const T* locations, unsigned dimension)  : _values(locations, locations + dimension)
	{
		assert(GetDimension() > 0);
	}

	/**  ...
	 * 
	 */
	BaseNode(const std::vector<unsigned>& locations) : _values(locations)
	{
		assert(GetDimension() > 0);
	}

	/**  ��ȡĳ�������ϵ�����ֵ.
	 * 
	 */
	T GetLocationValue(unsigned direction) const
	{
		assert(direction < GetDimension());
		return this->_values[direction];
	}
	/**  ����ĳ�������ϵ�����ֵ
	 * 
	 */
	void SetLocationValue(T value, unsigned direction)
	{
		assert(direction < GetDimension());
		this->_values[direction] = value;
	}

	/**  ��ȡ�ýڵ��ϵ�ά��
	 * 
	 */
	unsigned GetDimension() const
	{
		return static_cast<unsigned>(_values.size());
	}

private:
	std::vector<T> _values; 
};

} // namespace icesop


#endif // BASENODE_H
