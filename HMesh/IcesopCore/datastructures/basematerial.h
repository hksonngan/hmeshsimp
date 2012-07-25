/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 类BaseMaterial:
** 模板类,定义材料
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
// history : 修改Material的底层数据结构 用以同时支持脚标检索和关键字检索 by duguguiyu at 2008-06-17
// 增加材料ID、材料描述、材料类型三类属性 by LiShilei 09-07-22
//
////////////////////////////////////////////////////////////////

#ifndef BASEMATERIAL_H
#define BASEMATERIAL_H

#include <string>
#include <map>
#include <vector>
#include <assert.h>

namespace icesop {

/**  表征一种材料.
 *   一种材料有一个独一无二的名字，包含有若干个属性，每个属性可以有一个唯一的表示符号或名字。
 *   用户可以通过名字或脚标来获取相关的属性。
 *   @note 与结果类不同，该类采用统一的接口表示材料的属性，这是因为材料的属性太多无法统一，
 *         因此，为了维护其可理解性和统一性，需要做一个属性名的标准，否则，这个结构很难作为基础的数据结构。
 *   @note 不是每一个材料属性都需要名字，可以通过默认的序列直接用下标访问，这样做是为了兼顾效率和可理解性。
 *   @todo 尚不支持随温度性质发生改变的材料，需依照情况添加支持。
 *   @todo 要统一材料属性的命名（即，统一一个材料库）。
 *   @author dugugiyu
 *   @date 2008-06-16
 *   @ingroup Kernel
 */
template<typename T>
class BaseMaterial
{
public:
	typedef T Value;  ///< 内部值的存放类型

public:
	/**  构造函数.
	 *
	 */
	BaseMaterial() : _materialID(-1), _materialName(""), _materialType(0)
	{
	}

	/**  构造函数.
	 *
	 */
	BaseMaterial(int materialID, const std::string& materialName, const std::string& materialDescription, int materialType)
		: _materialID(materialID), _materialName(materialName), _materialDescription(materialDescription), _materialType(materialType)
	{
	}

	/**  是否是有效的材料ID.
	 *
	 */
	bool IsValidID()
	{
		return this->_materialID != -1;
	}

	/**  获得材料ID.
	 *
	 */
	int GetID()
	{
		return this->_materialID;
	}

	/**  获得材料名称.
	 *
	 */
	std::string GetName() const
	{
		return this->_materialName;
	}

	/**  获得材料描述.
	 *
	 */
	std::string GetDescription() const
	{
		return this->_materialDescription;
	}

	/**  获得材料类型.
	 *
	 */
	int GetType()
	{
		return this->_materialType;
	}

	/**  添加属性的名称和值.
	 *   @warning 如果属性名不唯一，新加的属性值将覆盖原有的。
	 */
	unsigned AddValue(const std::string & key, const Value value)
	{
		unsigned pos = AddValue(value);

		this->_keys[key] = pos;

		return pos;
	}

	/**  添加无名的属性.
	 *   @note 使用该接口添加的值，往往是有默认的次序，不需要通过下标来访问。
	 */
	unsigned AddValue(const Value value)
	{
		this->_values.push_back(value);

		return GetItemNumbers() - 1;
	}

	/**  是否包含某个属性.
	 *
	 */
	bool HaveItem(const std::string& key) const
	{
		return this->_keys.find(key) != this->_keys.end();
	}

	/**  获得某个属性在列表中的位置.
	 *   这个函数可以提供如下功能：
	 *   对于一系列连续的属性值，只有第一个具有描述，则可以获得第一个的位置，并顺序取出其他属性
	 */
	unsigned GetValueIndex(const std::string& key) const
	{
		assert(this->_keys.find(key) != this->_keys.end());

		return this->_keys.find(key)->second;
	}

	/**  获得总属性的数量.
	 *
	 */
	unsigned GetItemNumbers() const
	{
		return static_cast<unsigned>(this->_values.size());
	}

	/**  获得属性值.
	 *
	 */
	Value GetValue(const std::string& key) const
	{
		assert(this->_keys.find(key) != this->_keys.end());

		return _values[this->_keys.find(key)->second];
	}

	/**  获得指定位置的属性值.
	 *
	 */
	Value GetValue(const unsigned index) const
	{
		assert(index <= GetItemNumbers());

		return this->_values[index];
	}

	/**  清空.
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
