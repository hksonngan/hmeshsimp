#ifndef __MC_SIMP_TYPES_H__
#define __MC_SIMP_TYPES_H__

#include <string>
#include <boost/unordered_map.hpp>
#include "fnv1_inc.h"
#include "common_types.h"

using std::string;

//extern const char X_AXIS_EDGE;
//extern const char Y_AXIS_EDGE;
//extern const char Z_AXIS_EDGE;
//extern const char CUBE_VERTEX;
//
//class MCVertexIndex {
//public:
//	char type; // X_AXIS_EDGE, Y_AXIS_EDGE, Z_AXIS_EDGE, CUBE_VERTEX;
//	unsigned int x, y, z;
//
//public:
//	MCVertexIndex(char _t, unsigned int _x, unsigned int _y, unsigned int _z) {
//		type = _t;
//		x = _x;
//		y = _y;
//		z = _z;
//	}
//
//	bool operator == (MCVertexIndex &i) {
//		return type == i.type && x == i.x && y == i.y && z == i.z;
//	}
//};
//
//class MCVertIndexHash {
//	std::size_t operator() (MCVertexIndex const& vi) const {
//		ostringstream oss;
//		oss << vi.type << " " << vi.x << " " << vi.y << " " << vi.z;
//		hash::fnv_1a fnv;
//		return fnv(oss.str());
//	}
//};
//
//using boost::unordered::unordered_map;
//typedef unordered_map<MCVertexIndex, HVertex>

class VertexHash {
public:
	std::size_t operator() (HVertex const& v) const {
		char buf[sizeof(HVertex) + 1];
		buf[sizeof(HVertex)] = '\0';
		memcpy(buf, &v, sizeof(HVertex));

		string str(buf);
		hash::fnv_1a fnv;
		return fnv(str);
	}
};

using boost::unordered::unordered_map;
typedef unordered_map<HVertex, unsigned int, VertexHash> VertexIndexMap;

enum InterpOnWhich { None, Edge, Vert1, Vert2 };

#endif