/*
 *  Some Helper Functions for Debug
 *
 *  Author: Ht
 *  Email : waytofall916 at gmail dot com
 *
 *  Copyright (C) Ht. All rights reserved.
 */

#ifndef _PADAP_BEBUG_H_
#define _PADAP_BEBUG_H_

// for debug !!
void checkSolver(OctNode* node, int n) {
	int qem_count = 0, qem_out_count = 0, qem_not_solved_count = 0;
	//for (int i = 0; i < n; i ++) {
	//	if (node[i].child_count == 3) {
	//		qem_not_solved_count ++;
	//	} else if (node[i].child_count == 5) {
	//		qem_out_count ++;
	//	} else if (node[i].child_count == 7) {
	//		qem_count ++;
	//	}
	//}

	std::cout << "#qem count: " << qem_count << std::endl
		<< "#qem out count: " << qem_out_count << std::endl
		<< "#qem not solved count: " << qem_not_solved_count << std::endl << std::endl;
}

// for debug !!
void partitionCheck(OctFace *face, int zero_count, int count) {
	int i;

	for (i = 0; i < zero_count; i ++) {
		if (face[i].split_addr != 0) {
			cout << "#partition not passed" << endl << endl;
			return;
		}
	}

	for (; i < count; i ++) {
		if (face[i].split_addr == 0) {
			cout << "#partition not passed" << endl << endl;
			return;
		}
	}

	cout << "#partition passed" << endl << endl;
}

const std::string nodeIndexToStr(const unsigned int &index) {
	if (index == INVALID_NODE) 
		return "-";
	else {
		ostringstream oss;
		oss << index;
		return oss.str();
	}
}

void printArr(const OctFace *face_arr1, const OctFace *face_arr2, int count) {
	std::ofstream fout("face_array.txt");
	for (int i = 0; i < count;  i ++) {
		fout << (int)face_arr1[i].level1 << ":" << nodeIndexToStr(face_arr1[i].index1) << " "
			<< (int)face_arr1[i].level2 << ":" << nodeIndexToStr(face_arr1[i].index2) << " '" 
			<< (int)face_arr1[i].face_dir << "' #" << face_arr1[i].split_addr
			<< " > "
			<< (int)face_arr2[i].level1 << ":" << nodeIndexToStr(face_arr2[i].index1) << " "
			<< (int)face_arr2[i].level2 << ":" << nodeIndexToStr(face_arr2[i].index2) << " '"
			<< (int)face_arr2[i].face_dir << "' #" << face_arr2[i].split_addr
			<< std::endl;
	}
}

void printArr(const OctFace *face_arr, const int count, const char *filename) {
	std::ofstream fout(filename);
	for (int i = 0; i < count;  i ++) {
		fout << (int)face_arr[i].level1 << ":" << nodeIndexToStr(face_arr[i].index1) << " "
			<< (int)face_arr[i].level2 << ":" << nodeIndexToStr(face_arr[i].index2) << " '" 
			<< (int)face_arr[i].face_dir << "' #" << face_arr[i].split_addr
			<< std::endl;
	}
}

struct faceIndexComp {
	bool operator() (const OctFace &f1, const OctFace &f2) {
		if (f1.face_dir < f2.face_dir)
			return true;
		else if (f1.face_dir > f2.face_dir)
			return false;
		if (f1.level1 < f2.level1)
			return true;
		else if (f1.level1 > f2.level1)
			return false;
		if (f1.index1 < f2.index1)
			return true;
		else if (f1.index1 > f2.index1)
			return false;
		if (f1.level2 < f2.level2)
			return true;
		else if (f1.level2 > f2.level2)
			return false;
		if (f1.index2 < f2.index2)
			return true;

		return false;
	}
};

struct faceIndexComp2 {
	bool operator() (const OctFace &f1, const OctFace &f2) {
		if (f1.level1 < f2.level1)
			return true;
		else if (f1.level1 > f2.level1)
			return false;
		if (f1.index1 < f2.index1)
			return true;
		else if (f1.index1 > f2.index1)
			return false;
		if (f1.face_dir < f2.face_dir)
			return true;

		return false;
	}
};

void printPartialFace(const OctFace *face_arr, const int count, const char *filename) {
	std::ofstream fout(filename);
	OctFace *pface= new OctFace[count * 2];

	for (int i = 0; i < count;  i ++) {
		pface[i*2].level1 = face_arr[i].level1;
		pface[i*2].index1 = face_arr[i].index1;
		pface[i*2].level2 = face_arr[i].level2;
		pface[i*2].index2 = face_arr[i].index2;
		pface[i*2].face_dir = face_arr[i].face_dir;
		pface[i*2].split_addr = i;

		pface[i*2+1].level1 = face_arr[i].level2;
		pface[i*2+1].index1 = face_arr[i].index2;
		pface[i*2+1].level2 = face_arr[i].level1;
		pface[i*2+1].index2 = face_arr[i].index1;
		pface[i*2+1].face_dir = face_arr[i].face_dir;
		pface[i*2+1].split_addr = i;
	}

	std::sort(pface, pface + count*2, faceIndexComp2());

	for (int i = 0; i < count * 2; i ++) {
		fout << (int)pface[i].level1 << ":" << nodeIndexToStr(pface[i].index1) << " "
			<< (int)pface[i].level2 << ":" << nodeIndexToStr(pface[i].index2) << " '" 
			<< (int)pface[i].face_dir << "' #" << pface[i].split_addr
			<< std::endl;
	}

	delete[] pface;
}

void arrEqualCheck(OctFace *face_arr1, OctFace *face_arr2, int count) {
	std::sort(face_arr1, face_arr1 + count, faceIndexComp());
	std::sort(face_arr2, face_arr2 + count, faceIndexComp());

	for (int i = 0; i < count;  i ++) {
		if (face_arr1[i].level1 != face_arr2[i].level1 || 
			face_arr1[i].index1 != face_arr2[i].index1 || 
			face_arr1[i].level2 != face_arr2[i].level2 || 
			face_arr1[i].index2 != face_arr2[i].index2) {
			cout << "#array not equal" << endl << endl;
		}
	}

	cout << "#array equal" << endl << endl;
}

// for debug
void scanCheck(OctFace *face_arr1, OctFace *face_arr2, int count) {
	for (int i = 0; i < count - 1;  i ++) {
		if (face_arr2[i + 1].split_addr - face_arr2[i].split_addr != face_arr1[i].split_addr) {
			cout << "#face split count scan failed" << endl << endl;
			return;
		}
	}
}

void checkZeroTri(float *tri, int count) {
	int zero_count = 0;
	int i, j;
	for (i = 0; i < count; i ++) {
		bool zero = true;
		for (j = 0;  j < 9; j ++) 
			if (tri[i*9+j] != 0) {
				zero = false;
				break;
			}
			if (tri[i*9+j] != tri[i*9+j]) {
				int k = 0;
			}
			if (zero)
				zero_count ++;
	}

	std::cout << std::endl << "zero triangle count: " << zero_count << std::endl;
}

void checkTriOutBound(float *tri, int count) {
	int i, j;
	float x_min = cubeStart[maxDepth*3] *volSet.thickness.s[0];
	float x_max = (cubeStart[maxDepth*3] +cubeCount[maxDepth*3]) *volSet.thickness.s[0];
	float y_min = cubeStart[maxDepth*3+1] *volSet.thickness.s[1];
	float y_max = (cubeStart[maxDepth*3+1] +cubeCount[maxDepth*3+1]) *volSet.thickness.s[1];
	float z_min = cubeStart[maxDepth*3+2] *volSet.thickness.s[2];
	float z_max = (cubeStart[maxDepth*3+2] +cubeCount[maxDepth*3+2]) *volSet.thickness.s[2];

	for (i = 0; i < count*3; i ++) {
		if (tri[i*3] < x_min || tri[i*3] > x_max || 
			tri[i*3+1] < y_min || tri[i*3+1] > y_max || 
			tri[i*3+2] < z_min || tri[i*3+2] > z_max) {
			cout << "tri " << i/3 << " out of bound" << endl;
		}
	}
}

//void checkOctree(int start_depth) {
//	for (int d = start_depth; d <= maxDepth; d ++) {
//		for (int i = 0; i < level_count[d]; i ++) {
//			int x_start = (h_octLvlPtr[d][i].cube_index[0]* cubeSize[d*3] + 
//				cubeStart[d*3])* volSet.thickness.s[0];
//			int x_end = ((h_octLvlPtr[d][i].cube_index[0]+1)* cubeSize[d*3] + 
//				cubeStart[d*3])* volSet.thickness.s[0];
//			if (h_octLvlPtr[d][i].dual_vert.x)
//		}
//	}
//}

#ifdef __CUDA_DBG
void dbgBufCopyBack() {
	checkCudaErrors(cudaMemcpy(hdbg_buf, h_dbg_buf_devptr, 
		sizeof(hdbg_buf), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}
#endif

void chekNodeIndexOutBound(
	const int &edge_depth, const int &edge_index, const char &node_level, 
	const unsigned int &node_index, const int & start_depth, int node_num
){
	if (node_level < start_depth || node_level > maxDepth) {
		cout << "edge [" << edge_depth << ", " << edge_index << "]" << " node" << node_num <<
			" level out bound" << endl;
		return;
	}
	if (node_index != INVALID_NODE && node_index >= level_count[node_level]) {
		cout << "edge [" << edge_depth << ", " << edge_index << "]" << " node" << node_num <<
			" index out bound" << endl;
		return;
	}
}

void checkEdgeNodeIndexOutBound(int start_depth) {
	for (int depth = start_depth; depth <= maxDepth; depth ++) {
		for (int i = 0; i < n_medge[depth]; i ++) {
			OctEdge &e = h_medge_ptr[depth][i];
			chekNodeIndexOutBound(depth, i, e.level1, e.index1, start_depth, 1);
			chekNodeIndexOutBound(depth, i, e.level2, e.index2, start_depth, 2);
			chekNodeIndexOutBound(depth, i, e.level3, e.index3, start_depth, 3);
			chekNodeIndexOutBound(depth, i, e.level4, e.index4, start_depth, 4);
		}
	}
}

#endif //_PADAP_BEBUG_H_