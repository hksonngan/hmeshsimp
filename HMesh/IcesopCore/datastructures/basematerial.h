/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ��BaseMaterial:
** ģ����,�������
**
** Author : shan @2011
**
****************************************************************************/

////////////////////////////////////////////////////////////////
//
// author: dugugiyu
// date  : 2008-06-16
//
// description:
//
// history : �޸�Material�ĵײ����ݽṹ ����ͬʱ֧�ֽű�����͹ؼ��ּ��� by duguguiyu at 2008-06-17
// ���Ӳ���ID���������������������������� by LiShilei 09-07-22
//
////////////////////////////////////////////////////////////////

#ifndef BASEMATERIAL_H
#define BASEMATERIAL_H

#include <string>
#include <map>
#include <vector>
#include <assert.h>

namespace icesop {

/**  ����һ�ֲ���.
 *   һ�ֲ�����һ����һ�޶������֣����������ɸ����ԣ�ÿ�����Կ�����һ��Ψһ�ı�ʾ���Ż����֡�
 *   �û�����ͨ�����ֻ�ű�����ȡ��ص����ԡ�
 *   @note �����಻ͬ���������ͳһ�Ľӿڱ�ʾ���ϵ����ԣ�������Ϊ���ϵ�����̫���޷�ͳһ��
 *         ��ˣ�Ϊ��ά���������Ժ�ͳһ�ԣ���Ҫ��һ���������ı�׼����������ṹ������Ϊ���������ݽṹ��
 *   @note ����ÿһ���������Զ���Ҫ���֣�����ͨ��Ĭ�ϵ�����ֱ�����±���ʣ���������Ϊ�˼��Ч�ʺͿ�����ԡ�
 *   @todo �в�֧�����¶����ʷ����ı�Ĳ��ϣ�������������֧�֡�
 *   @todo Ҫͳһ�������Ե�����������ͳһһ�����Ͽ⣩��
 *   @author dugugiyu
 *   @date 2008-06-16
 *   @ingroup Kernel
 */
template<typename T>
class BaseMaterial
{
public:
	typedef T Value;  ///< �ڲ�ֵ�Ĵ������

public:
	/**  ���캯��.
	 *
	 */
	BaseMaterial() : _materialID(-1), _materialName(""), _materialType(0)
	{
	}

	/**  ���캯��.
	 *
	 */
	BaseMaterial(int materialID, const std::string& materialName, const std::string& materialDescription, int materialType)
		: _materialID(materialID), _materialName(materialName), _materialDescription(materialDescription), _materialType(materialType)
	{
	}

	/**  �Ƿ�����Ч�Ĳ���ID.
	 *
	 */
	bool IsValidID()
	{
		return this->_materialID != -1;
	}

	/**  ��ò���ID.
	 *
	 */
	int GetID()
	{
		return this->_materialID;
	}

	/**  ��ò�������.
	 *
	 */
	std::string GetName() const
	{
		return this->_materialName;
	}

	/**  ��ò�������.
	 *
	 */
	std::string GetDescription() const
	{
		return this->_materialDescription;
	}

	/**  ��ò�������.
	 *
	 */
	int GetType()
	{
		return this->_materialType;
	}

	/**  ������Ե����ƺ�ֵ.
	 *   @warning �����������Ψһ���¼ӵ�����ֵ������ԭ�еġ�
	 */
	unsigned AddValue(const std::string & key, const Value value)
	{
		unsigned pos = AddValue(value);

		this->_keys[key] = pos;

		return pos;
	}

	/**  �������������.
	 *   @note ʹ�øýӿ���ӵ�ֵ����������Ĭ�ϵĴ��򣬲���Ҫͨ���±������ʡ�
	 */
	unsigned AddValue(const Value value)
	{
		this->_values.push_back(value);

		return GetItemNumbers() - 1;
	}

	/**  �Ƿ����ĳ������.
	 *
	 */
	bool HaveItem(const std::string& key) const
	{
		return this->_keys.find(key) != this->_keys.end();
	}

	/**  ���ĳ���������б��е�λ��.
	 *   ������������ṩ���¹��ܣ�
	 *   ����һϵ������������ֵ��ֻ�е�һ����������������Ի�õ�һ����λ�ã���˳��ȡ����������
	 */
	unsigned GetValueIndex(const std::string& key) const
	{
		assert(this->_keys.find(key) != this->_keys.end());

		return this->_keys.find(key)->second;
	}

	/**  ��������Ե�����.
	 *
	 */
	unsigned GetItemNumbers() const
	{
		return static_cast<unsigned>(this->_values.size());
	}

	/**  �������ֵ.
	 *
	 */
	Value GetValue(const std::string& key) const
	{
		assert(this->_keys.find(key) != this->_keys.end());

		return _values[this->_keys.find(key)->second];
	}

	/**  ���ָ��λ�õ�����ֵ.
	 *
	 */
	Value GetValue(const unsigned index) const
	{
		assert(index <= GetItemNumbers());

		return this->_values[index];
	}

	/**  ���.
	 *
	 */
	void Clear()
	{
		this->_keys.clear();
		this->_values.clear();
	}

private:
	std::string _materialName;
	std::vector<Value> _values;
	std::map<std::string, unsigned> _keys;

	//new added properties LiShilei
	int _materialID;
	std::string _materialDescription;
	int _materialType;
};

} // namespace icesop

#endif // BASEMATERIAL_H
