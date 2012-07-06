#include "pcol_iterative.h"

CollapsableVertex PairCollapse::cvert;
CollapseFace PairCollapse::cface;
vert_arr PairCollapse::starVerts1;
vert_arr PairCollapse::starVerts2;

PairCollapse::PairCollapse() {

}

void PairCollapse::allocVerts(uint _vert_count) {
	vertices.resize(_vert_count);
}

void PairCollapse::allocFaces(uint _face_count) {
	faces.resize(_face_count);
}

void PairCollapse::intialize() {
	valid_verts = vertices.count();
	valid_faces = faces.count();
}