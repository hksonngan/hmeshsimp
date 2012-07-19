/*
 *  Vertex data structures for edge collapse
 *  For more detail, please refer to 'ECOL_DESIGN'
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 */


#ifndef __PCOL_VERTEX__
#define __PCOL_VERTEX__

#include <list>
#include <climits>

#include "util_common.h"
#include "h_dynamarray.h"
#include "pcol_other_structures.h"
#include "MixKit/MxQMetric3.h"


#define UNREFER			UCHAR_MAX		// interior unreferred vertex mark
#define REFERRED		UCHAR_MAX - 1	// interior referred vertex mark
/* below used when simplifying a patch of a mesh */
#define INTERIOR_BOUND	UCHAR_MAX - 2	// interior boundary (must be referred) vertex mark
#define EXTERIOR		UCHAR_MAX - 3	// exterior boundary (must be referred) vertex mark

using std::list;

typedef list<uint> face_list;
typedef HDynamArray<CollapsablePair*> pair_arr;
typedef HDynamArray<uint> face_arr;
typedef HDynamArray<uint> vert_arr;
//typedef HQEMatrix<float> q_matrix;
typedef MxQuadric3 q_matrix;

/* out-of-core version */
class CollapsedVertex: public HVertex {
public:
	uint	new_id;		// the id after the collapse
	uint	output_id;	// the id of the output model
	uchar	mark;
	/// I think it's useless!!!
	//HVertex	new_vertex;	// the new vertex after the collapse
						// used for collapsing sequence file

public:
	CollapsedVertex() { markv(UNREFER); };
	void setNewId(uint _id) { new_id = _id; }
	void markv(uchar m) { mark = m; }
	bool valid(uint v_index) { return v_index == new_id; }
	bool unreferred() { return mark == UNREFER; }
	bool interior() { return mark == UNREFER || mark == REFERRED; }
	bool interior_bound() { return mark == INTERIOR_BOUND; }
	bool exterior() { return mark == EXTERIOR; }
};

/* in-core version */
class CollapsableVertex: public CollapsedVertex {
public:
	inline CollapsableVertex();
	void allocAdjacents(uint faces_count = DFLT_STARS, uint pairs_count = DFLT_STARS) {
		adjacent_col_pairs.resize(pairs_count);
		adjacent_faces.resize(faces_count);
	}

public:

	/* The linked collapsable pairs
	 * in the heap, used for update.
	 * If it is edge collapse, it
	 * would be 'adjacent collapsable 
	 * edges'. The adjacent pairs should
	 * be updated with the collapse
	 * operation
	 * One of the element is NULL means
	 * that it has been decimated during
	 * the collapse of the corresponding
	 * vertex */ 
	pair_arr	adjacent_col_pairs;

	/* use when decimated based on the 
	 * face count, remove the face if 
	 * needed. the adjacent faces should
	 * be updated with the collapse operation */
	face_arr	adjacent_faces;

	// assisting variable use for linkage
	// information operation
	//short	flag;

private:
	// default star faces & pairs count
	static const uint DFLT_STARS = 6; 
};

CollapsableVertex::CollapsableVertex():
adjacent_faces(0),
adjacent_col_pairs(0)
{

}

/* edge collapse vertex with quadric error matrix */
class QuadricVertex: public CollapsableVertex {
public:

public:
	q_matrix	quadrics;	// quadric error matrix
};

class HierarchyVertex: public CollapsableVertex {
public:
	face_list	alter_faces;	// faces need to alter when the vertex expand/contract
	face_list	removed_faces;	// faces need to remove/insert when the vertex expand/contract
};

class HierarchyQuadricVertex: public HierarchyVertex {
public:
	q_matrix	quadrics;	// quadric error matrix
};

#endif //__PCOL_VERTEX__