/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ��BaseIndexs:
** ģ����,��������,BaseIndexes?
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

/**  һ�������������ṹ.
 *   ����һ���������ݽṹ����������ĳһ���ؼ��ʵĵ����������ýṹ�в���ž����ֵ�������ֵ���±ꡣ
 *   @todo ����Ҫ���һЩ�ӿڣ�����ClearValues�������ڲ��ĵ㵫���������ļ�ֵ��
 *   @author dugugiyu
 *   @date 2008-06-17
 *   @ingroup Kernel
 */
template<typename KeyValueType>
class BaseIndexs
{
public:
	typedef KeyValueType KeyValue;  ///< ��������.

	typedef std::vector<unsigned> Indexs;  ///<  ��������
	typedef std::map< KeyValue , Indexs > Keys;  ///<  ������������

public:
	/**  ���һ����.
	 *
	 */
	void AddKey(const KeyValue key)
	{
		this->_keys[key];
	}

	/**  ���һ������.
	 *
	 */
	void AddIndex(const KeyValue key, const unsigned index)
	{
		this->_keys[key].push_back(index);
	}

	/**  ���һ����������AddIndex��ͬ���ǣ�������ֵ�����Ѿ�����.
	 *
	 */
	void AddIndexToExistedKey(const KeyValue key, const unsigned index)
	{
		assert(HaveTheKey(key));

		this->_keys.find(key)->second.push_back(index);
	}

	/**  �Ƿ����һ����ֵ.
	 *
	 */
	bool HaveTheKey(const KeyValue key) const
	{
		return this->_keys.find(key) != this->_keys.end();
	}


	/**  ����ܵļ�ֵ��Ŀ.
	 *
	 */
	unsigned GetKeysNumber() const
	{
		return static_cast<unsigned>(this->_keys.size());
	}

	/**  ���������ֵ�ԵĿ�ʼλ��.
	 *
	 */
	typename Keys::const_iterator GetKeyBegin() const
	{
		return this->_keys.begin();
	}

	/**  ���������ֵ�ԵĽ���λ��.
	 *
	 */
	typename Keys::const_iterator GetKeyEnd() const
	{
		return this->_keys.end();
	}

	/**  ���ĳ������Ӧֵ���ϵĿ�ʼλ��.
	 *
	 */
	typename Indexs::const_iterator GetIndexsBegin(const KeyValue key) const
	{
		assert(this->_keys.find(key) != _keys.end());

		return this->_keys.find(key)->second.begin();
	}

	/**  ���ĳ������Ӧֵ���ϵĽ���λ��.
	 *
	 */
	typename Indexs::const_iterator GetIndexsEnd(const KeyValue key) const
	{
		assert(this->_keys.find(key) != _keys.end());

		return this->_keys.find(key)->second.end();
	}

	/**  ���������ֵ�ԵĽ���λ��.
	 *
	 */
	unsigned GetIndexsNumber(const KeyValue key) const
	{
		assert(this->_keys.find(key) != _keys.end());

		return static_cast<unsigned>(this->_keys.find(key)->second.size());
	}

	/**  ���.
	 *
	 */
	void Clear()
	{
		_keys.clear();
	}

	/**  ������ݣ���������ֵ.
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
