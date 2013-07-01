/*
 *  Vertex data structures for edge collapse
 *  For more detail, please refer to 'ECOL_DESIGN'
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */


#ifndef __PCOL_VERTEX__
#define __PCOL_VERTEX__

#include <list>
#include <climits>

#include "common_types.h"
#include "h_dynamarray.h"
#include "pcol_other_structures.h"
#include "MixKit/MxQMetric3.h"


#define UNREFER         0    // interior unreferred vertex mark
#define REFERRED        1    // interior referred vertex mark
/* below used when simplifying a patch of a mesh */
#define INTERIOR_BOUND  2    // interior boundary (must be referred) vertex mark
#define EXTERIOR        3    // exterior boundary (must be referred) vertex mark
/* below used for incremental simplifying */
#define UNFINAL         4    // unfinalized vertices
#define FINAL           5    // finalized vertices (the succeeding faces won't reference the vertex)

using std::list;

typedef list<uint> face_list;
typedef HDynamArray<CollapsablePair*> pair_arr;
typedef HDynamArray<uint> face_arr;
typedef HDynamArray<uint> vert_arr;
//typedef HQEMatrix<float> q_matrix;
//typedef MxQuadric3 q_matrix;

// a vertex type used for pair collapse
/* out-of-core version ?? */
class CollapsedVertex: public HVertex {
public:
	uint	new_id;		// the id after the collapse
	uint	output_id;	// the id of the output model
	uchar	mark;
	//HVertex	new_vertex;	// the new vertex after the collapse
						// used for collapsing sequence file

public:
	CollapsedVertex() { markv(UNREFER); };
	void setNewId(uint _id) { new_id = _id; }
	void setOutId(uint _id) { output_id = _id; }
	void markv(uchar m) { mark = m; }
	void unfinal() { mark = UNFINAL; }
	void finalize() { mark = FINAL; }

	bool valid(uint v_index) const { return v_index == new_id; } /* valid referred to uncollapsed */
	bool unreferred() const { return mark == UNREFER; }
	bool interior() const { return mark != EXTERIOR; }
	bool interior_bound() const { return mark == INTERIOR_BOUND; }
	bool exterior() const { return mark == EXTERIOR; }
	bool finalized() const { return mark == FINAL; }
};

// a vertex type used for pair collapse with linkage information
/* in-core version ?? */
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
	 * operation.
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

// unused
class HierarchyVertex: public CollapsableVertex {
public:
	face_list	alter_faces;	// faces need to alter when the vertex expand/contract
	face_list	removed_faces;	// faces need to remove/insert when the vertex expand/contract
};

#endif //__PCOL_VERTEX__