#ifndef MESH_GLOBAL_H
#define MESH_GLOBAL_H

#include <Qt/qglobal.h>

#ifdef MESH_LIB
# define MESH_EXPORT Q_DECL_EXPORT
#else
# define MESH_EXPORT Q_DECL_IMPORT
#endif

#endif // MESH_GLOBAL_H
