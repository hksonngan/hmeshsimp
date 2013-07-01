///////////////////////////////////////////////////////////
//  mesh_generator.cpp
//  Implementation of the Class MeshGenerator
//  Created on:      05-三月-2013 13:31:57
//  Original author: Edgar, Houtao
///////////////////////////////////////////////////////////

#define __HT_DLL_EXPORT__
#include "mesh_generator.h"
#include "mc_simp.h"
#include "trivial.h"
#include <string>

using std::string;

MeshGenerator::MeshGenerator(){

}

MeshGenerator::~MeshGenerator(){

}

// 网格生成
bool MeshGenerator::GenerateMesh(
    const int* sizes,                 // 三维体数据大小，三个成员，x，y，z 
    const float* spacings,            // 每层厚度，三个成员 
    const int* sample_stride,         // x, y, z方向上的采样宽度，三个成员
    const short* data,                // 体数据指针 
    const double iso_value,           // 等值面的值
    vector< float >& point_position   // 输出网格中顶点，x y z值连续存储，三个顶点为一个三角形
    ) {
    point_position.clear();
    VolumeSet vol_set;
    vol_set.volumeSize.s[0] = sizes[0];
    vol_set.volumeSize.s[1] = sizes[1];
    vol_set.volumeSize.s[2] = sizes[2];
    vol_set.thickness.s[0] = spacings[0];
    vol_set.thickness.s[1] = spacings[1];
    vol_set.thickness.s[2] = spacings[2];
    vol_set._data = reinterpret_cast<Byte *>(const_cast<short *>(data));
    vol_set.DATA_ARR_ALLOC_IN_THIS_OBJECT = false;
    vol_set.fileEndian = getSystemEndianMode();
    vol_set.format = DATA_SHORT;
    vol_set.formatSize = 2;
    vol_set.layeredRead = false;
    vol_set.cursor.s[0] = vol_set.cursor.s[1] = vol_set.cursor.s[2] = 0;
    MCSimp mcsimp;
    if (!mcsimp.genIsosurfaces(string(""), iso_value, 
        const_cast<int *>(sample_stride), point_position, &vol_set))
        return false;

    ofstream fout("gensimp.log", ofstream::app | ofstream::out);
    cout << mcsimp.info();
    fout << mcsimp.info();

    return true;
}

// 网格生成中使用边收缩简化
bool MeshGenerator::GenerateCollapse(
    const int* sizes,                 // 三维体数据大小，三个成员，x，y，z 
    const float* spacings,            // 每层厚度，三个成员 
    const int* sample_stride,         // x, y, z方向上体素的采样宽度，三个成员
    const short* data,                // 体数据指针 
    const double iso_value,           // 等值面的值
    const double decimate_rate,       // 简化率
    vector< float >& point_position,  // 输出网格中顶点，x y z值连续存储 
    vector< int >& triangle_index,    // 输出网格中三角形索引，三个索引连在一块儿	
    const unsigned int buffer_size    // 简化过程中缓存的大小，这个值越大简化质量越高，但速度越慢
    ) {
    point_position.clear();
    triangle_index.clear();
    VolumeSet vol_set;
    vol_set.volumeSize.s[0] = sizes[0];
    vol_set.volumeSize.s[1] = sizes[1];
    vol_set.volumeSize.s[2] = sizes[2];
    vol_set.thickness.s[0] = spacings[0];
    vol_set.thickness.s[1] = spacings[1];
    vol_set.thickness.s[2] = spacings[2];
    vol_set._data = reinterpret_cast<Byte *>(const_cast<short *>(data));
    vol_set.DATA_ARR_ALLOC_IN_THIS_OBJECT = false;
    vol_set.fileEndian = getSystemEndianMode();
    vol_set.format = DATA_SHORT;
    vol_set.formatSize = 2;
    vol_set.layeredRead = false;
    vol_set.cursor.s[0] = vol_set.cursor.s[1] = vol_set.cursor.s[2] = 0;

    MCSimp mcsimp;
    unsigned int numvert, numface;

    //unsigned int vol_size = vol_set.volumeSize.s[0] * vol_set.volumeSize.s[1]* 
    //    vol_set.volumeSize.s[2];

    if (!mcsimp.genCollapse(string(""), iso_value, decimate_rate, 
        const_cast<int *>(sample_stride), buffer_size, numvert, numface, &vol_set)) {
        cerr << "#error occurred during simplification" << endl << endl;
        return false;
    }

	point_position.resize(numvert * 3);
	triangle_index.resize(numface * 3);
	mcsimp.toIndexedMesh(point_position, triangle_index);

    ofstream fout("gensimp.log", ofstream::app | ofstream::out);
    cout << mcsimp.info();
    fout << mcsimp.info();

    return true;
}