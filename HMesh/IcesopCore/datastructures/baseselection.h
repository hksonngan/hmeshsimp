/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 类BaseSelection:
** 模板类,定义选择集
**
** Author : shan @2011
**
****************************************************************************/

////////////////////////////////////////////////////////////////
//
// author: dugugiyu
// date  : 2008-01-21
//
// description:
//
// history :
//
////////////////////////////////////////////////////////////////

#ifndef BASESELECTION_H
#define BASESELECTION_H

#include "BaseCollection.h"
#include "BaseSubElement.h"
#include <vector>

#include "assert.h"

namespace Fea
{
	namespace Data
	{
		/**  表示选取单元、点的集合.
		 *   该结构中只包含所选取点的信息，当用户传入有限元单元集合的时候，该类可以根据有限元单元本身蕴涵的连接性信息，提取出选取的有限元单元。
		 *
		 *   @author dugugiyu
		 *   @date 2008-01-21
		 *   @ingroup Kernel
		 */
		class BaseSelection
		{
		public:
			/** 判定该节点是否已被添加到选择集中.
			 *
			 */
			bool IsSelected(unsigned nodeIndex) const
			{
				for(std::vector<unsigned>::const_iterator iter = this->selectedNodeIndexs.begin(); iter != this->selectedNodeIndexs.end(); iter++)
				{
					if((*iter) == nodeIndex)
						return true;
				}

				return false;
			}

			/** 将节点添加到选择集中.
			 *  这里采用最简单的添加方式，不对重复节点进行判断，即不调用IsSelected方法。
			 *  相关逻辑交给上层来维护，如果上层采取的是可能重复的插入算法，需要先调用IsSelected函数进行判断，再调用该函数进行插入。
			 *  如此算法能够支持上层做算法优化。
			 */
			void AddNodeIndex(unsigned nodeIndex)
			{
				this->selectedNodeIndexs.push_back(nodeIndex);
			}

			/** 泛型方法，获得所选择的子单元集合.
			 *  其中，传入的Element通常是BaseElement的实现类，支持GetNodeNumber()、GetNodeIndex(unsigned)方法。
			 *  @warning 生成节点的SubElement（NodeNumber == 1），不能调用该方法，而是调用GetSelectedNodeIndexs方法。
			 */
			template< typename Element > std::vector<BaseSubElement> GetSelectedElementIndexs(
				const BaseCollection<Element> * elements, unsigned nodeNumber) const
			{
				assert(elements != NULL);
				std::vector<BaseSubElement> selectedSubElements;
				/*
				 * 采用的基本算法是遍历element集合，判定每个element有多少个node已被选中
				 * 如果该element被选中的节点个数等于要求的子单元（SubElement）的节点数量，则添加该SubElement作为选中SubElement
				 */

				unsigned elementsNumber = elements->Size();
				for(unsigned elementIndex = 0; elementIndex < elementsNumber; elementIndex++)
				{
					Element currentElement = elements->GetValue(elementIndex);
					std::vector<unsigned> currentSelectedNodes;
					for(unsigned pos = 0; pos < currentElement.GetNodeNumber(); pos++)
					{
						if(this->IsSelected(currentElement.GetNodeIndex(pos)))
						{
							currentSelectedNodes.push_back(currentElement.GetNodeIndex(pos));
						}
					}

					if(static_cast<unsigned>(currentSelectedNodes.size()) == nodeNumber)
					{
						selectedSubElements.push_back(BaseSubElement(currentSelectedNodes));
					}
				}

				return selectedSubElements;
			}

			/**  获得选取点的集合.
			 *
			 */
			std::vector<BaseSubElement> GetSelectedNodeIndexs() const
			{
				std::vector<BaseSubElement> nodeElements;

				for(std::vector<unsigned>::const_iterator iter = selectedNodeIndexs.begin(); iter != selectedNodeIndexs.end(); iter++)
				{
					nodeElements.push_back(BaseSubElement(std::vector<unsigned>(1, (*iter))));
				}

				return nodeElements;
			}

		private:
			std::vector<unsigned> selectedNodeIndexs;
		};

	} // end namespace data
} // end namespace fea


#endif // BASESELECTION_H
