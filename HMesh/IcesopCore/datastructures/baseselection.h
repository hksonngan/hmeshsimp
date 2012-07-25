/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** ��BaseSelection:
** ģ����,����ѡ��
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
		/**  ��ʾѡȡ��Ԫ����ļ���.
		 *   �ýṹ��ֻ������ѡȡ�����Ϣ�����û���������Ԫ��Ԫ���ϵ�ʱ�򣬸�����Ը�������Ԫ��Ԫ�����̺�����������Ϣ����ȡ��ѡȡ������Ԫ��Ԫ��
		 *
		 *   @author dugugiyu
		 *   @date 2008-01-21
		 *   @ingroup Kernel
		 */
		class BaseSelection
		{
		public:
			/** �ж��ýڵ��Ƿ��ѱ���ӵ�ѡ����.
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

			/** ���ڵ���ӵ�ѡ����.
			 *  ���������򵥵���ӷ�ʽ�������ظ��ڵ�����жϣ���������IsSelected������
			 *  ����߼������ϲ���ά��������ϲ��ȡ���ǿ����ظ��Ĳ����㷨����Ҫ�ȵ���IsSelected���������жϣ��ٵ��øú������в��롣
			 *  ����㷨�ܹ�֧���ϲ����㷨�Ż���
			 */
			void AddNodeIndex(unsigned nodeIndex)
			{
				this->selectedNodeIndexs.push_back(nodeIndex);
			}

			/** ���ͷ����������ѡ����ӵ�Ԫ����.
			 *  ���У������Elementͨ����BaseElement��ʵ���֧࣬��GetNodeNumber()��GetNodeIndex(unsigned)������
			 *  @warning ���ɽڵ��SubElement��NodeNumber == 1�������ܵ��ø÷��������ǵ���GetSelectedNodeIndexs������
			 */
			template< typename Element > std::vector<BaseSubElement> GetSelectedElementIndexs(
				const BaseCollection<Element> * elements, unsigned nodeNumber) const
			{
				assert(elements != NULL);
				std::vector<BaseSubElement> selectedSubElements;
				/*
				 * ���õĻ����㷨�Ǳ���element���ϣ��ж�ÿ��element�ж��ٸ�node�ѱ�ѡ��
				 * �����element��ѡ�еĽڵ��������Ҫ����ӵ�Ԫ��SubElement���Ľڵ�����������Ӹ�SubElement��Ϊѡ��SubElement
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

			/**  ���ѡȡ��ļ���.
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
