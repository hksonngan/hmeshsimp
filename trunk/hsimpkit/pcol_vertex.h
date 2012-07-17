/*
 *  Vertex data structures for edge collapse
 *  For more detail, please refer to 'ECOL_DESIGN'
 *
 *  Author: Ht
 *  Email : waytofall916@gmail.com
 *
 *  Copyright (C) Ht-waytofall. All rights reserved.
 *	
 *  This file is part of hmeshsimp.
 *
 *  hmeshsimp is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  hmeshsimp is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with hmeshsimp.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef __PCOL_VERTEX__
#define __PCOL_VERTEX__

#include <list>
#include <climits>

#include "util_common.h"
#include "h_dynamarray.h"
#include "pcol_other_structures.h"
#include "MixKit/MxQMetric3.h"

#define INVALID_VERT UINT_MAX

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
	/// I think it's useless!!!
	HVertex	new_vertex;	// the new vertex after the collapse
						// used for collapsing sequence file

public:
	void setNewId(uint _id) { new_id = _id; }
	bool valid(uint v_index) { return v_index == new_id; }
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