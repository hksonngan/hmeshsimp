/*
 *	vertex cluster and some concerning classes
 *	author: ht
 *  email : waytofall916@gmail.com
 */

#include "vertex_cluster.h"
#include <iostream>
#include <fstream>


/* --HVertexCluster-- */

HVertex& HVertexCluster::
			calcRepresentativeVertex(RepCalcPolicy p, float c_max_x, float c_min_x, float c_max_y, 
										float c_min_y, float c_max_z, float c_min_z) {

	/* if the qem is not invertible, 
	   it won't modify the input vertex */
	if (p == QEM_INV) {
		HVertex new_vertex;
		qem.calcRepresentativeVertex(new_vertex);

		// the vertex falls in the cluster
		if (representative_vertex != new_vertex &&
			new_vertex.x >= c_min_x && new_vertex.x <= c_max_x &&
			new_vertex.y >= c_min_y && new_vertex.y <= c_max_y &&
			new_vertex.z >= c_min_z && new_vertex.z <= c_max_z) {

			representative_vertex = new_vertex;
		}
	}

	return representative_vertex;
}

/* --- HVertexClusterContainer --- */

bool HVertexClusterContainer::create(Integer _x_partition, Integer _y_partition, Integer _z_partition)
{
	x_partition = _x_partition; y_partition = _y_partition;
	z_partition = _z_partition;

	valid_clusters = 0;
	cluster_count = x_partition * y_partition * z_partition;

	pp_cluster = new HVertexCluster*[cluster_count];
	// initialize null
	memset((void*)pp_cluster, 0, sizeof(HVertexCluster*) * cluster_count);

	return true;
}

bool HVertexClusterContainer::clear()
{
	Integer i, j, k;

	for (i = 0; i < x_partition; i ++)
		for (j = 0; j < y_partition; j ++)
			for (k = 0; k < z_partition; k ++)
			{
				HVertexCluster* p_cluster = get(i, j, k);
				if (p_cluster)
				{
					delete p_cluster;
				}
			}

	delete[] pp_cluster;
	x_partition = 0; y_partition = 0;
	z_partition = 0;
	cluster_count = 0;

	return true;
}

// add a face qem to the cluster, create a cluster if the it doesn't exist
bool HVertexClusterContainer::addFace(Integer i, Integer j, Integer k, HSoupTriangle tri)
{
	HVertexCluster* p_cluster = 
		pp_cluster[i * y_partition * z_partition + j * z_partition + k];
	
	if (p_cluster == NULL)
	{
		p_cluster = new HVertexCluster;
		pp_cluster[i * y_partition * z_partition + j * z_partition + k] = p_cluster;
		valid_clusters ++;
	}

	// add the qem of the triangle to the cluster
	HQEMatrix<float> new_qem(tri);
	(p_cluster->getQemRef()) += (new_qem);

	return true;
}

void HVertexClusterContainer::addVertex(Integer i, Integer j, Integer k, HVertex vertex)
{
	HVertexCluster* p_cluster = 
		pp_cluster[i * y_partition * z_partition + j * z_partition + k];

	if (p_cluster == NULL) {
		p_cluster = new HVertexCluster;
		pp_cluster[i * y_partition * z_partition + j * z_partition + k] = p_cluster;
		valid_clusters ++;
	}

	p_cluster->addVertex(vertex);
}

void HVertexClusterContainer::generateIndexForClusters()
{
	Integer i, j, k, c = 0;
	HVertexCluster* p_cluster;

	for (i = 0; i < x_partition; i ++)
		for (j = 0; j < y_partition; j ++)
			for (k = 0; k < z_partition; k ++)
			{
				p_cluster = pp_cluster[i * y_partition * z_partition + j * z_partition + k];

				if (p_cluster) {
					p_cluster->setVIndex(c);
					c ++;
				}
			}
}

void HVertexClusterContainer::calcAllRepresentativeVertices(RepCalcPolicy p)
{
	Integer i, j, k, c = 0;
	HVertexCluster* p_cluster;

	for (i = 0; i < x_partition; i ++)
		for (j = 0; j < y_partition; j ++)
			for (k = 0; k < z_partition; k ++)
			{
				p_cluster = get(i, j, k);

				if (p_cluster) {
					//p_cluster->calcRepresentativeVertex(p);
				}
			}
}


/* --- HVertexClusterSimp --- */

HVertexClusterSimp::HVertexClusterSimp() {
	x_partition = 0; y_partition = 0;
	z_partition = 0;

	max_x = min_x = 0; max_y = min_y = 0;
	max_z = min_z = 0;
	x_slice = 0; y_slice = 0;
	z_slice = 0;

	face_set.rehash(INIT_BUCKET_SIZE);
}

HVertexClusterSimp::~HVertexClusterSimp()
{
	std::cout << "\thash set bucket count: " << face_set.bucket_count() << std::endl
		<< "\taverage elements per bucket: " << face_set.load_factor() << std::endl
		<< "\tmaximum load factor: " << face_set.max_load_factor() << std::endl; 
}

bool HVertexClusterSimp::create(Integer _x_partition, Integer _y_partition, Integer _z_partition, RepCalcPolicy _p) {
	x_partition = _x_partition; y_partition = _y_partition; 
	z_partition = _z_partition;
	vertex_clusters.create(x_partition, y_partition, z_partition);
	rep_calc_policy = _p;

	return true;
}

void HVertexClusterSimp::setBoundBox(float _max_x, float _min_x, float _max_y, float _min_y, float _max_z, float _min_z) {

	float half_range_x = (_max_x - _min_x) / 2 * 1.025;
	float half_range_y = (_max_y - _min_y) / 2 * 1.025;
	float half_range_z = (_max_z - _min_z) / 2 * 1.025;
	max_x = (_max_x + _min_x) / 2 + half_range_x;
	min_x = (_max_x + _min_x) / 2 - half_range_x;
	max_y = (_max_y + _min_y) / 2 + half_range_y;
	min_y = (_max_y + _min_y) / 2 - half_range_y;
	max_z = (_max_z + _min_z) / 2 + half_range_z;
	min_z = (_max_z + _min_z) / 2 - half_range_z;
}

void HVertexClusterSimp::clear() 
{
	vertex_clusters.clear();
	face_set.clear();

	x_partition = 0; y_partition = 0;
	z_partition = 0;

	max_x = min_x = 0; max_y = min_y = 0;
	max_z = min_z = 0;
	x_slice = 0; y_slice = 0;
	z_slice = 0;
}

HTripleIndex<Integer> HVertexClusterSimp::retrieveIndex(HVertex v)
{
	HTripleIndex<Integer> i;

	i.i = (int)((v.x - min_x) / x_slice);
	if (i.i >= x_partition) {
		i.i = x_partition - 1;
	}
	i.j = (int)((v.y - min_y) / y_slice);
	if (i.j >= y_partition) {
		i.j = y_partition - 1;
	}
	i.k = (int)((v.z - min_z) / z_slice);
	if (i.k >= z_partition) {
		i.k = z_partition - 1;
	}

	return i;
}

bool HVertexClusterSimp::addSoupTriangle(HSoupTriangle triangle)
{
	// index for three vertices in clusters
	HTripleIndex<Integer> i1, i2, i3;

	if (x_slice == 0) {
		x_slice = (max_x - min_x) / x_partition;
		y_slice = (max_y - min_y) / y_partition;
		z_slice = (max_z - min_z) / z_partition;
	}

	// retrieve cluster index for each vertex
	i1 = retrieveIndex(triangle.v1);
	i2 = retrieveIndex(triangle.v2);
	i3 = retrieveIndex(triangle.v3);

	// add quadric error metrics of the surface defined by the triangle 
	// to each different clusters referred by the triangle
	vertex_clusters.addFace(i1, triangle);

	if (i2 != i1) {
		vertex_clusters.addFace(i2, triangle);
	}

	if (i3 != i1 && i3 != i2) {
		vertex_clusters.addFace(i3, triangle);
	}

	vertex_clusters.addVertex(i1, triangle.v1);
	vertex_clusters.addVertex(i2, triangle.v2);
	vertex_clusters.addVertex(i3, triangle.v3);

	// add the triangular face indexing different clusters
	// if the three indices refer to different clusters
	if (i1 != i2 && i1 != i3 && i2 != i3) {
		face_set.insert(HFaceIndex(i1, i2, i3));
	}

	return true;
}

bool HVertexClusterSimp::generateIndexedMesh()
{
	vertex_clusters.generateIndexForClusters();

	return true;
}

const char* getPolicyStr(RepCalcPolicy p)
{
	switch(p)
	{
	case QEM_INV: 
		return "qem diff";
	case MEAN_VERTEX: 
		return "mean vertex";
	default: 
		return NULL;
	}

	return NULL;
}

void HVertexClusterSimp::getClusterRange(HTripleIndex<Integer> index, float &_max_x, float &_min_x, 
										 float &_max_y, float &_min_y, float &_max_z, float &_min_z)
{
	_min_x = min_x + index.i * x_slice;
	_max_x = _min_x + x_slice;
	_min_y = min_y + index.j * y_slice;
	_max_y = _min_y + y_slice;
	_min_z = min_z + index.k * z_slice;
	_max_z = _min_z + z_slice;
}

bool HVertexClusterSimp::writeToPly(char* filename)
{
	std::ofstream fout(filename);
	if (fout.bad()) {
		return false;
	}

	/* write head */
	fout << "ply" << std::endl;
	fout << "format ascii 1.0" << std::endl;
	fout << "comment generated by ht out-of-core simplification" << std::endl;
	fout << "comment representative vertex calculating policy: " << getPolicyStr(rep_calc_policy) << std::endl;
	fout << "comment triangular faces only" << std::endl;
	fout << "comment may contain non-referenced vertex" << std::endl;

	fout << "element vertex " << vertex_clusters.getValidClusterCount() << std::endl;
	fout << "property float x" << std::endl;
	fout << "property float y" << std::endl;
	fout << "property float z" << std::endl;
	fout << "element face " << face_set.size() << std::endl;
	fout << "property list uchar int vertex_indices" << std::endl;
	fout << "end_header" << std::endl;

	/* write vertices */
	Integer i, j, k, c = 0;
	HVertexCluster* p_cluster;
	float cmaxx, cminx, cmaxy, cminy, cmaxz, cminz;

	for (i = 0; i < x_partition; i ++)
		for (j = 0; j < y_partition; j ++)
			for (k = 0; k < z_partition; k ++)
			{
				p_cluster = vertex_clusters.get(i, j, k);

				if (p_cluster) {
					p_cluster->setVIndex(c);

					// calculating representative vertex
					getClusterRange(HTripleIndex<Integer>(i, j, k), cmaxx, cminx, cmaxy, cminy, cmaxz, cminz);
					p_cluster->calcRepresentativeVertex(rep_calc_policy, cmaxx, cminx, cmaxy, cminy, cmaxz, cminz);

					// write
					fout << p_cluster->getRepresentativeVertex().x << " "
						<< p_cluster->getRepresentativeVertex().y << " "
						<< p_cluster->getRepresentativeVertex().z << std::endl;
					c ++;
				}
			}

	/* write faces */
	HFaceIndexSet::iterator iter;

	for (iter = face_set.begin(); iter != face_set.end(); iter ++)
	{
		if ((*iter).v1CIndex != (*iter).v2CIndex && (*iter).v1CIndex != (*iter).v3CIndex && (*iter).v2CIndex != (*iter).v3CIndex)
		{

			i = vertex_clusters.get((*iter).v1CIndex)->getVIndex();
			j = vertex_clusters.get((*iter).v2CIndex)->getVIndex();
			k = vertex_clusters.get((*iter).v3CIndex)->getVIndex();
			// write
			fout << "3 " << i << " " << j << " " << k << std::endl;
		}
	}

	// statistics
	std::cout << "\t#info:" << std::endl << "\twrite simplified mesh successfully"
		<< "\tfile name: " << filename << std::endl
		<< "\tvertex count: " << c << std::endl
		<< "\tface count: " <<  face_set.size() << std::endl
		<< "\trep vertex policy: " << getPolicyStr(rep_calc_policy) << std::endl;

	return true;
}