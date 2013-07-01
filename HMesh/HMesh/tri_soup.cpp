/*
	triangle soup container
	author: ht
*/

#include "tri_soup.h"

TriangleSoupContainer::TriangleSoupContainer()
	:init_size(50000),
	p_tri_soup(NULL),
	tri_soup_size(0),
	tri_soup_count(0)
{
}

TriangleSoupContainer::~TriangleSoupContainer()
{
	if (p_tri_soup)
	{
		delete[] p_tri_soup;
	}
}

void TriangleSoupContainer::init()
{
	if (tri_soup_size == 0)
	{
		p_tri_soup = new SoupTriangle[init_size];
		tri_soup_size = init_size;
		tri_soup_count = 0;
	}
}

void TriangleSoupContainer::clear()
{
	if (p_tri_soup)
	{
		delete[] p_tri_soup;
	}

	p_tri_soup = NULL;
	tri_soup_size = 0;
	tri_soup_count = 0;
}

bool TriangleSoupContainer::read(const char* filename)
{
	if (sstream.openForRead(filename) == 0)
		return false;

	init();
	int i = 0;

	try {
		for (; true; tri_soup_count ++)
		{
			if (sstream.readNext() == 0)
			{
				break;
			}
		
			if (tri_soup_count >= tri_soup_size)
			{
				SoupTriangle *new_soup = new SoupTriangle[tri_soup_size * 2];
				memcpy(new_soup, p_tri_soup, tri_soup_size * sizeof(float) * 9);
				delete[] p_tri_soup;
				p_tri_soup = new_soup;
				tri_soup_size *=  2;
			}

			p_tri_soup[tri_soup_count].vert1.x = sstream.getFloat(0, 0);
			p_tri_soup[tri_soup_count].vert1.y = sstream.getFloat(0, 1);
			p_tri_soup[tri_soup_count].vert1.z = sstream.getFloat(0, 2);
			p_tri_soup[tri_soup_count].vert2.x = sstream.getFloat(1, 0);
			p_tri_soup[tri_soup_count].vert2.y = sstream.getFloat(1, 1);
			p_tri_soup[tri_soup_count].vert2.z = sstream.getFloat(1, 2);
			p_tri_soup[tri_soup_count].vert3.x = sstream.getFloat(2, 0);
			p_tri_soup[tri_soup_count].vert3.y = sstream.getFloat(2, 1);
			p_tri_soup[tri_soup_count].vert3.z = sstream.getFloat(2, 2);

		}
	} catch (std::ios_base::failure f) {
		std::cerr << "read file exception" << std::endl;
		return false;
	}

	sstream.closeForRead();

	return true;
}

SoupTriangle TriangleSoupContainer::operator()(unsigned int i)
{
	return p_tri_soup[i];
}

unsigned int TriangleSoupContainer::count()
{
	return tri_soup_count;
}

TriSoupStream* TriangleSoupContainer::getTriSStream()
{
	return &sstream;
}