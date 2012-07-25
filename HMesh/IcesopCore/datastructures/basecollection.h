/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ��BaseCollection:
** ģ����,���弯��
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

/**  ����Ԫϵͳ���õĻ��������࣬�ṩ�򵥵Ĵ���ӿں��������ڹ�������.
 *   �ü���Ӧ�ñ�׼��vector����й�����װ�˲��ֵ�vector�ӿڡ�
 *   ���׼��vector��ͬ���������Զ����ڴ����ģ�͡�����������ֻ���ܻ���ֵ���͵�ģ�����͡�
 *
 *   @todo ������ò���Ҫ�ڸ����ڴ����Ĳ��֣�Ӧ�ðѸ�������ͨ��std::vector����������Ҫ��ӵĻ�����Ҫ��һ�����Ƹ���Ľӿ�
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
