/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 类BaseIndexs:
** 模板类,定义索引,BaseIndexes?
**
** Author : shan @2011
**
****************************************************************************/

////////////////////////////////////////////////////////////////
//
// author: dugugiyu
// date  : 2008-06-17
//
// description:
//
// history :
//
////////////////////////////////////////////////////////////////

#ifndef BASEINDEXS_H
#define BASEINDEXS_H

#include <map>
#include <vector>

#include <assert.h>

namespace icesop {

/**  一个辅助的索引结构.
 *   这是一个辅助数据结构，帮助建立某一个关键词的倒排索引，该结构中不存放具体的值，仅存放值的下标。
 *   @todo 还需要添加一些接口，比如ClearValues，消除内部的点但保留索引的键值。
 *   @author dugugiyu
 *   @date 2008-06-17
 *   @ingroup Kernel
 */
template<typename KeyValueType>
class BaseIndexs
{
public:
	typedef KeyValueType KeyValue;  ///< 键的类型.

	typedef std::vector<unsigned> Indexs;  ///<  索引集合
	typedef std::map< KeyValue , Indexs > Keys;  ///<  健和索引集合

public:
	/**  添加一个健.
	 *
	 */
	void AddKey(const KeyValue key)
	{
		this->_keys[key];
	}

	/**  添加一个索引.
	 *
	 */
	void AddIndex(const KeyValue key, const unsigned index)
	{
		this->_keys[key].push_back(index);
	}

	/**  添加一个索引，与AddIndex不同的是，该索引值必须已经存在.
	 *
	 */
	void AddIndexToExistedKey(const KeyValue key, const unsigned index)
	{
		assert(HaveTheKey(key));

		this->_keys.find(key)->second.push_back(index);
	}

	/**  是否存在一个键值.
	 *
	 */
	bool HaveTheKey(const KeyValue key) const
	{
		return this->_keys.find(key) != this->_keys.end();
	}


	/**  获得总的键值数目.
	 *
	 */
	unsigned GetKeysNumber() const
	{
		return static_cast<unsigned>(this->_keys.size());
	}

	/**  获得整个键值对的开始位置.
	 *
	 */
	typename Keys::const_iterator GetKeyBegin() const
	{
		return this->_keys.begin();
	}

	/**  获得整个键值对的结束位置.
	 *
	 */
	typename Keys::const_iterator GetKeyEnd() const
	{
		return this->_keys.end();
	}

	/**  获得某个键对应值集合的开始位置.
	 *
	 */
	typename Indexs::const_iterator GetIndexsBegin(const KeyValue key) const
	{
		assert(this->_keys.find(key) != _keys.end());

		return this->_keys.find(key)->second.begin();
	}

	/**  获得某个键对应值集合的结束位置.
	 *
	 */
	typename Indexs::const_iterator GetIndexsEnd(const KeyValue key) const
	{
		assert(this->_keys.find(key) != _keys.end());

		return this->_keys.find(key)->second.end();
	}

	/**  获得整个键值对的结束位置.
	 *
	 */
	unsigned GetIndexsNumber(const KeyValue key) const
	{
		assert(this->_keys.find(key) != _keys.end());

		return static_cast<unsigned>(this->_keys.find(key)->second.size());
	}

	/**  清空.
	 *
	 */
	void Clear()
	{
		_keys.clear();
	}

	/**  清除数据，仅保留键值.
	 *
	 */
	void ClearData()
	{
		for(typename Keys::iterator iter = _keys.begin();
			iter != _keys.end();
			++iter)
		{
			iter->second.clear();
		}
	}

private:
	Keys _keys;
};

} // namespace icesop

#endif // BASEINDEXS_H
