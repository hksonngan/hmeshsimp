/****************************************************************************
**
** Copyright (C) 2011-2014 CG&CAD School of Software, Tsinghua University
**
** 定义ICESOP中基本的通用变量、函数和类
**
** Author : shan @2011
**
****************************************************************************/

#ifndef ICESOP_COMMON_H
#define ICESOP_COMMON_H

namespace icesop {

enum OperateMode {
	OPERATEMODE_NONE, 		//have no operation, or clear the operation mode.
    OPERATEMODE_SELECT,
	OPERATEMODE_PAN,
	OPERATEMODE_ROTATE,
	OPERATEMODE_ZOOM,

	OPERATEMODE_SELECTION,
	OPERATEMODE_INSERTCONNECTION,
	OPERATEMODE_INSERTINPUTVAR,
	OPERATEMODE_INSERTOUTPUTVAR,
	OPERATEMODE_INSERTFILE,
	OPERATEMODE_INSERTACTION
};

enum ZoomMode {
	ZOOMMODE_NONE,
	ZOOMMODE_CLICK,
	ZOOMMODE_DRAG
};

enum MdiType {
	MDITYPE_WORKFLOW,
	MDITYPE_MODEL,
	MDITYPE_MESH
};

} // namespace icesop

#endif // ICESOP_COMMON_H
