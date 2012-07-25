/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 类BaseCollection:
** 模板类,定义集合
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
// history :
//
////////////////////////////////////////////////////////////////

#ifndef BASECOLLECTION_H
#define BASECOLLECTION_H

#include <vector>
#include <assert.h>
#include <ostream>

namespace icesop {

/**  有限元系统所用的基础集合类，提供简单的处理接口和生命周期管理能力.
 *   该集合应用标准的vector库进行管理，封装了部分的vector接口。
 *   与标准的vector不同，该类有自动的内存管理模型【待建】，并只接受基于值类型的模板类型。
 *
 *   @todo 如果觉得不需要在附加内存管理的部分，应该把该类用普通的std::vector替代，如果需要添加的话，需要进一步完善该类的接口
 *   @author dugugiyu
 *   @date 2007-12-29
 *   @ingroup Kernel
 */
template < typename Value >
class BaseCollection
{
public:
	typedef Value ValueType;

public:
	BaseCollection()
	{
	}
	BaseCollection(unsigned totalIndexs) : values(std::vector<Value>(totalIndexs))
	{
	}

public:
	unsigned AddValue(Value value)
	{
		values.push_back(value);
		return static_cast<unsigned>(values.size() - 1);
	}
	void SetValue(Value value, unsigned index)
	{
		assert(index < Size());
		values[index] = value;
	}
	Value GetValue(unsigned index) const
	{
		//assert(index < Size());
		return values[index];
	}

	const Value & GetConstValue(unsigned index) const
	{
		return values[index];
	}

	void Merge(const std::vector<Value> & newValues)
	{
		values.insert(values.end(), newValues.begin(), newValues.end());
	}

	bool Empty() const
	{
		return values.empty();
	}
	unsigned Size() const
	{
		return static_cast<unsigned>(values.size());
	}
	void Clear()
	{
		values.clear();
	}

private:
	std::vector<Value> values;
};

} // namespace icesop

#endif // BASECOLLECTION_H
