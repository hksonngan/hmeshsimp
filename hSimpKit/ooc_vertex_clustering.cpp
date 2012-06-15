#include "ooc_vertex_clustering.h"

bool HOOCVertexClustering::run(int x_partition, int y_partition, int z_partition, RepCalcPolicy p,
	char* inputfilename, char* outputfilename)
{
	TriSoupStream sstream;
	if (sstream.openForRead(inputfilename) == 0) {
		std::cerr << "#error: open triangle soup file failed" << std::endl;
		return false;
	}

	HVertexClusterSimp vcsimp;
	HSoupTriangle soup_tri;

	vcsimp.setBoundBox(sstream.getMaxX(), sstream.getMinX(), sstream.getMaxY(), 
		sstream.getMinY(), sstream.getMaxZ(), sstream.getMinZ());
	vcsimp.create(x_partition, y_partition, z_partition, p);

	while (sstream.readNext())
	{
		// retrieve the triangle soup
		soup_tri.v1.set(sstream.getFloat(0, 0), sstream.getFloat(0, 1), sstream.getFloat(0, 2));
		soup_tri.v2.set(sstream.getFloat(1, 0), sstream.getFloat(1, 1), sstream.getFloat(1, 2));
		soup_tri.v3.set(sstream.getFloat(2, 0), sstream.getFloat(2, 1), sstream.getFloat(2, 2));

		vcsimp.addSoupTriangle(soup_tri);
	}

	if (vcsimp.writeToPly(outputfilename) == false) {
		std::cerr << "#error: write to ply file failed" << std::endl;
		return false;
	}

	return true;
}