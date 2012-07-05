/*
 *  vertex data structures for edge collapse
 *  for detail, please refer to 'ecol_design.txt'
 *
 *  author: ht
 *  email : waytofall916@gmail.com
 */

#ifndef __PCOL_VERTEX__
#define __PCOL_VERTEX__

#include <list>

#include "util_common.h"
#include "h_dynamarray.h"
#include "pcol_pair.h"

unsigned std::list;

typedef list<uint> face_list;
typedef HDynamArray<CollapsablePair*> pair_arr;
typedef HDynamArray<uint> face_arr;
typedef HQEMatrix<float> q_matrix;
typedef HDynamArray<uint> vert_arr;

/* out-of-core version */
class CollapsedVertex: public HVertex {
public:
	uint	new_id;		// the id after the collapse
	uint	output_id;	// the id of the output model
	/// temporarily deprecated, can't quite figure
	/// the use of it, if it's really useful. may
	/// need some further design. I thought that
	/// the new_vertex may not need to be explicitly
	/// stored, just update the x, y, z filed will ok.
	//HVertex	new_vertex;	// the new vertex after the collapse
};

/* in-core version */
class CollapsableVertex: public CollapsedVertex {
public:
	inline CollapsableVertex();

public:

	// the linked collapsable pairs
	// in the heap, used for update
	// if it is edge collapse, it
	// would be 'adjacent collapsable 
	// edges'
	pair_arr	adjacent_col_pairs;
	// use when decimated based on the 
	// face count, remove the face if 
	// needed
	face_arr	adjacent_faces;

private:
	// the face star count is set to 6
	static const uint INIT_ADJACENT_FACES_COUNT = 6; 
};

CollapsableVertex::CollapsableVertex():
adjacent_faces(INIT_ADJACENT_FACES_COUNT),
adjacent_col_pairs(INIT_ADJACENT_FACES_COUNT)
{

}

/* edge collapse vertex with quadric error matrix */
class QuadricVertex: public CollapsableVertex {
public:

public:
	q_matrix	quadrics;	// quadric error matrix
}

class HierarchyVertex: public CollapsableVertex {
public:
	face_list	alter_faces;	// faces need to alter when the vertex expand/contract
	face_list	removed_faces;	// faces need to remove/insert when the vertex expand/contract
}

class HierarchyQuadricVertex: public HierarchyVertex {
public:
	q_matrix	quadrics;	// quadric error matrix
}

#endif //__PCOL_VERTEX__