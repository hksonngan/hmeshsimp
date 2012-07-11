/*************************************
 author   : houtao
 spec     : some additional functions
            for PLY manipulating, especially
			some augment for the memory
			usage
 email    : waytofall916@gmail.com
 copyleft : basically GPL
 date     : 29/5/2012
*************************************/

#include "ply_inc.h"
#include "stdlib.h"

void clean_ply()
{
	if (vert_other)
	{
		free(vert_other);
		vert_other = NULL;
	}
	if (face_other)
	{
		free(face_other);
		face_other = NULL;
	}
	if (other_elements)
	{
		free(other_elements);
		other_elements = NULL;
	}

	if (flist)
	{
		for (int i = 0; i < nfaces; i ++)
		{
			if (flist[i].verts)
			{
				free(flist[i].verts);
			}
		}
		free (flist);
		flist = NULL;
	}

	if(vlist)
	{
		for (int i = 0; i < nverts; i ++)
		{
			if (vlist[i].other_props)
			{
				free(vlist[i].other_props);
			}
		}
		free (vlist);
		vlist = NULL;
	}


	nverts = 0;
	nfaces = 0;
}