///////////////////////////////////////////////////////////
//  mesh_generator.cpp
//  Implementation of the Class MeshGenerator
//  Created on:      05-����-2013 13:31:57
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

// ��������
bool MeshGenerator::GenerateMesh(
    const int* sizes,                 // ��ά�����ݴ�С��������Ա��x��y��z 
    const float* spacings,            // ÿ���ȣ�������Ա 
    const int* sample_stride,         // x, y, z�����ϵĲ�����ȣ�������Ա
    const short* data,                // ������ָ�� 
    const double iso_value,           // ��ֵ���ֵ
    vector< float >& point_position   // ��������ж��㣬x y zֵ�����洢����������Ϊһ��������
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

// ����������ʹ�ñ�������
bool MeshGenerator::GenerateCollapse(
    const int* sizes,                 // ��ά�����ݴ�С��������Ա��x��y��z 
    const float* spacings,            // ÿ���ȣ�������Ա 
    const int* sample_stride,         // x, y, z���������صĲ�����ȣ�������Ա
    const short* data,                // ������ָ�� 
    const double iso_value,           // ��ֵ���ֵ
    const double decimate_rate,       // ����
    vector< float >& point_position,  // ��������ж��㣬x y zֵ�����洢 
    vector< int >& triangle_index,    // ���������������������������������һ���	
    const unsigned int buffer_size    // �򻯹����л���Ĵ�С�����ֵԽ�������Խ�ߣ����ٶ�Խ��
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