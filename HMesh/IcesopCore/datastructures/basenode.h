/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 类BaseNode:
** 模板类,定义节点
** 通过定义vector保存dimension(维度)个变量
** 可以对各维度数据进行操作
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
// history : 移除力和约束相关的接口 by duguguiyu at 2008-01-18
//           移除力和约束的相关变量 by duguguiyu at 2008-02-25
//           改为运行时确定维度 by duguguiyu at 2008-05-23
//           将一些int类型的数改为unsigned类型的 by duguguiyu at 2008-05-24
//       
////////////////////////////////////////////////////////////////

#ifndef BASENODE_H
#define BASENODE_H

#include <vector>
#include <assert.h>

namespace icesop {

/**  表示网格中的一个节点.
 *   @todo 需添加测试
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

	/**  获取某个方向上的坐标值.
	 * 
	 */
	T GetLocationValue(unsigned direction) const
	{
		assert(direction < GetDimension());
		return this->_values[direction];
	}
	/**  设置某个方向上的坐标值
	 * 
	 */
	void SetLocationValue(T value, unsigned direction)
	{
		assert(direction < GetDimension());
		this->_values[direction] = value;
	}

	/**  获取该节点上的维度
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
