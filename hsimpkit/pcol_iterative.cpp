#include "pcol_iterative.h"
#include <iostream>
#include <sstream>
#include <string>
#include "ply_stream.h"

using std::ostringstream;
using std::endl;

PairCollapse::PairCollapse() {
	faceIndexComp.setFaces(&faces);

#ifdef _VERBOSE
	merge_face_count = 1;
	fverbose.open("verbose");
#endif
}

PairCollapse::~PairCollapse() {
	cvert.adjacent_col_pairs.setNULL();
	cvert.adjacent_faces.setNULL();
}

void PairCollapse::allocVerts(uint _vert_count) {
#if ARRAY_USE == ARRAY_NORMAL
	vertices.resize(_vert_count);
#endif
}

void PairCollapse::allocFaces(uint _face_count) {
#if ARRAY_USE == ARRAY_NORMAL
	faces.resize(_face_count);
#endif
}

void PairCollapse::addVertex(const HVertex& vert) {
	cvert.Set(vert.x, vert.y, vert.z);
	cvert.markv(UNREFER);

#if ARRAY_USE == ARRAY_NORMAL
	cvert.setNewId(vertices.count());
	cvert.setOutId(vertices.count());
	vertices.push_back(cvert);
	v(vertices.count() - 1).allocAdjacents(DFLT_STAR_FACES, DFLT_STAR_PAIRS);
#else
	uint id = vertices.size();
	cvert.setNewId(id);
	cvert.setOutId(id);
	std::pair<ECVertexMap::iterator, bool> _pair = vertices.insert(ECVertexMap::value_type(id, cvert));
	((_pair.first)->second).allocAdjacents(DFLT_STAR_FACES, DFLT_STAR_PAIRS);
#endif
}

void PairCollapse::addVertex(const uint& index, const HVertex& vert, uchar mark) {
	cvert.Set(vert.x, vert.y, vert.z);
	cvert.markv(mark);

#if ARRAY_USE == ARRAY_NORMAL
	cvert.setNewId(index);
	cvert.setOutId(index);
	vertices.push_back(cvert);
	v(vertices.count() - 1).allocAdjacents(DFLT_STAR_FACES, DFLT_STAR_PAIRS);
#else
	cvert.setNewId(index);
	cvert.setOutId(index);
	std::pair<ECVertexMap::iterator, bool> _pair = vertices.insert(ECVertexMap::value_type(index, cvert));
	((_pair.first)->second).allocAdjacents(DFLT_STAR_FACES, DFLT_STAR_PAIRS);
#endif
}

bool PairCollapse::addFace(const HFace &face) {
	//face.sortIndex();
	cface.set(face.i, face.j, face.k);

	if (!cface.valid()) {
		addInfo("#ERROR: duplicate verts in input face\n");
		return false;
	}
#if ARRAY_USE == ARRAY_NORMAL
	if (!cface.indicesInRange(0, vertices.count() - 1)) {
		addInfo("#ERROR: vertex out of range in input face\n");
		return false;
	}
#endif

	uint id;
#if ARRAY_USE == ARRAY_NORMAL
	id = faces.count();
	faces.push_back(cface);
#else
	id = faces.size();
	faces.insert(ECFaceMap::value_type(id, cface));
#endif

	// add the face index to the vertices
	v(face.i).adjacent_faces.push_back(id);
	v(face.j).adjacent_faces.push_back(id);
	v(face.k).adjacent_faces.push_back(id);

	// set the new_id field
	v(face.i).markv(REFERRED);
	v(face.j).markv(REFERRED);
	v(face.k).markv(REFERRED);

	return true;
}

void PairCollapse::addCollapsablePair(CollapsablePair *new_pair) {
	v(new_pair->vert1).adjacent_col_pairs.push_back(new_pair);
	v(new_pair->vert2).adjacent_col_pairs.push_back(new_pair);

	if (!new_pair->is_in_heap()) {
		pair_heap.insert(new_pair);
	}
}

void PairCollapse::initialize() {
	initValids();
	unreferVertsCheck();
	collectPairs();
}

void PairCollapse::initValids() { 
#if ARRAY_USE == ARRAY_NORMAL
	valid_verts = vertices.count(); 
	valid_faces = faces.count(); 
#else
	valid_verts = vertices.size();
	valid_faces = faces.size();
#endif

#ifdef _VERBOSE
	another_valid_faces = valid_faces;
#endif
}

void PairCollapse::collapsePair(pCollapsablePair pair) {
	uint vert1 = pair->vert1, vert2 = pair->vert2;

	// set the new_id field and new_vertex field in order 
	// to invalidate vert2 and maintain the collapse footprint
	v(vert2).setNewId(vert1);
	//vertices[pair->vert2].new_vertex.Set(pair->new_vertex);
	// vert1 will be the collapsed vertex, set to the new position
	v(vert1).Set(pair->new_vertex);
	valid_verts --;

	mergePairs(vert1, vert2);
	mergeFaces(vert1, vert2);
#if ARRAY_USE == ARRAY_USE_HASH
	vertices.erase(vert2);
#endif
}

bool PairCollapse::targetVert(uint target_count) {
	ostringstream ostr;
#if ARRAY_USE == ARRAY_USE_HASH
	ostr << "\tfaces buckets: " << faces.bucket_count() << endl
		<< "\tavg num of faces per bucket: " << faces.load_factor() << endl
		<< "\tmax load factor: " << faces.max_load_factor() << endl
		<< "\tverts buckets: " << vertices.bucket_count() << endl
		<< "\tavg num of verts per bucket: " << vertices.load_factor() << endl
		<< "\tmax load factor: " << vertices.max_load_factor() << endl << endl;
#endif

	CollapsablePair* top_pair;
	run_time.setStartPoint();
	while(valid_verts > target_count) {
		top_pair = (CollapsablePair *)pair_heap.extract();
		if (!top_pair)
			break;
		collapsePair(top_pair);
	}
	run_time.setEndPoint();

	ostr << "\tmodel simplified" << endl
		<< "\tverts:\t" << valid_verts << "\tfaces:\t" << valid_faces << endl
		<< "\ttime consuming:\t" << run_time.getElapseStr() << endl << endl;
	addInfo(ostr.str());

	return true;
}

bool PairCollapse::targetFace(uint target_count) {
	ostringstream ostr;
#if ARRAY_USE == ARRAY_USE_HASH
	ostr << "\tfaces buckets: " << faces.bucket_count() << endl
		<< "\tavg num of faces per bucket: " << faces.load_factor() << endl
		<< "\tmax load factor: " << faces.max_load_factor() << endl
		<< "\tverts buckets: " << vertices.bucket_count() << endl
		<< "\tavg num of verts per bucket: " << vertices.load_factor() << endl
		<< "\tmax load factor: " << vertices.max_load_factor() << endl << endl;
#endif

	CollapsablePair* top_pair;
	run_time.setStartPoint();
	while(valid_faces > target_count) {
		top_pair = (CollapsablePair *)pair_heap.extract();
		if (!top_pair)
			break;
		collapsePair(top_pair);
	}
	run_time.setEndPoint();

	ostr << "\tmodel simplified" << endl
		<< "\tverts:\t" << valid_verts << "\tfaces:\t" << valid_faces << endl
		<< "\ttime consuming:\t" << run_time.getElapseStr() << endl << endl;
	addInfo(ostr.str());

	return true;
}

bool PairCollapse::readPly(char* filename) {
	PlyStream plyStream;
	int i;
	HVertex v;
	HFace f;

	read_time.setStartPoint();
	if (!plyStream.openForRead(filename)) {

		ostringstream oss;
		oss << "\t#ERROR: open file " << filename << " failed" << endl;
		addInfo(oss.str());
		return false;
	}

	this->allocVerts(plyStream.getVertexCount());
	this->allocFaces(plyStream.getFaceCount());

	cvert.adjacent_col_pairs.setNULL();
	cvert.adjacent_faces.setNULL();

	for (i = 0; i < plyStream.getVertexCount(); i ++) {
		if (!plyStream.nextVertex(v)) 
			return false;
		addVertex(v);
	}

	for (i = 0; i < plyStream.getFaceCount(); i ++) {
		if (!plyStream.nextFace(f)) 
			return false;
		addFace(f);
	}

	initialize();
	read_time.setEndPoint();
	ostringstream ostr;

	ostr << "\tread complete" << endl
		<< "\tfile name:\t" << filename << endl
		<< "\treferred verts:\t" << valid_verts << "\tfaces:\t" << plyStream.getFaceCount() << endl
		<< "\tread time:\t" << read_time.getElapseStr() << endl << endl;

	addInfo(ostr.str());

	return true;
}

bool PairCollapse::writePly(char* filename) {
	ofstream fout(filename);
	if (fout.bad())
		return false;

	write_time.setStartPoint();

	/* write head */
	fout << "ply" << endl;
	fout << "format ascii 1.0" << endl;
	fout << "comment generated by ht pair collapse" << endl;

	fout << "element vertex " << valid_verts << endl;
	fout << "property float x" << endl;
	fout << "property float y" << endl;
	fout << "property float z" << endl;
	fout << "element face " << valid_faces << endl;
	fout << "property list uchar int vertex_indices" << endl;
	fout << "end_header" << endl;

	uint valid_vert_count = 0;
	//for (i = 0; i < vertices.count(); i ++)
	//	if (vertices[i].valid(i) && !vertices[i].unreferred()) {
	//		fout << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << endl;
	//		vertices[i].output_id = valid_vert_count;
	//		valid_vert_count ++;
	//	}
	_for_loop(vertices, ECVertexMap) {
		CollapsableVertex& cvert = _retrieve_elem(vertices);	
		if (cvert.valid(_retrieve_index()) && !cvert.unreferred()) {
			fout << cvert.x << " " << cvert.y << " " << cvert.z << endl;
			cvert.output_id = valid_vert_count;
			valid_vert_count ++;
		}
	}

	uint valid_face_count = 0;
	//for (i = 0; i < faces.count(); i ++) 
	//	if (faces[i].valid()) {
	//		fout << "3 " << vertices[faces[i].i].output_id << " "
	//			<< vertices[faces[i].j].output_id << " "
	//			<< vertices[faces[i].k].output_id << endl;
	//		valid_face_count ++;
	//	}
	_for_loop(faces, ECFaceMap) {
		CollapsableFace& cface = _retrieve_elem(faces);
		if (cface.valid()) {
			fout << "3 " << v(cface.i).output_id << " "
				<< v(cface.j).output_id << " "
				<< v(cface.k).output_id << endl;
			valid_face_count ++;
		}
	}

	// statistics
	write_time.setEndPoint();
	ostringstream ostr;

	ostr << "\tsimplified mesh written" << endl
		<< "\tfile name:\t" << filename << endl
		<< "\twrite time:\t" << write_time.getElapseStr() << endl 
		<< "\ttrue valid verts: " << valid_vert_count << endl 
		<< "\ttrue valid faces: " << valid_face_count << endl
		<< "\tvert map size: " << vertices.size() << endl
		<< "\tface map size: " << faces.size() << endl
		<< endl;

	addInfo(ostr.str());

	return true;
}

void PairCollapse::outputIds(char* filename) {
	ofstream fout(filename);

	fout << "id\tnew_id\tout_put" << endl;

	//for (int i = 0; i < vertices.count(); i ++) {
	//	fout << i << "\t" << vertices[i].new_id << "\t" << vertices[i].output_id << endl;
	//}
	_for_loop(vertices, ECVertexMap) {
		CollapsableVertex& cvert = _retrieve_elem(vertices);
		fout << _retrieve_index() << "\t" << cvert.new_id << "\t" << cvert.output_id << endl;
	}
}

void PairCollapse::generateOutputId() {
	uint valid_vert_count = 0;

	//for (int i = 0; i < vertices.count(); i ++)
	//	if (vertices[i].valid(i)) {
	//		vertices[i].output_id = valid_vert_count;
	//		valid_vert_count ++;
	//	}
	_for_loop(vertices, ECVertexMap) {
		CollapsableVertex& cvert = _retrieve_elem(vertices);
		if (cvert.valid(_retrieve_index())) {
			cvert.output_id = valid_vert_count;
			valid_vert_count ++;
		}
	}
}

///////////////////////////////////////////////////////////////
// operations other than simplification

void PairCollapse::addInfo(std::string s) { info += s; }

void PairCollapse::totalTime() {
	HAugTime total_time = read_time + run_time + write_time;
	
	ostringstream ostr;
	ostr << "\ttotal time:\t" << total_time.getElapseStr() << endl << endl;
	addInfo(ostr.str());
}