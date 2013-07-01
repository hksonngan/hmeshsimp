///////////////////////////////////////////////////////////
//  mesh_generator.h
//  Implementation of the Class MeshGenerator
//  Created on:      05-����-2013 13:31:57
//  Original author: Edgar, Houtao
///////////////////////////////////////////////////////////

#ifndef _ORTHO_SYS_MESH_GENERATOR_H_
#define _ORTHO_SYS_MESH_GENERATOR_H_

#include <vector>

#ifdef __HT_DLL_EXPORT__
#define Dll_INTERFACE __declspec( dllexport )
#else
#define Dll_INTERFACE __declspec( dllimport )
#endif

using std::vector;

class Dll_INTERFACE MeshGenerator {
public:
	MeshGenerator();
	~MeshGenerator();
	
	// ��������
	bool GenerateMesh(
		const int* sizes,                 // ��ά�����ݴ�С��������Ա��x��y��z 
		const float* spacings,            // ÿ���ȣ�������Ա 
		const int* sample_stride,         // x, y, z�����ϵĲ�����ȣ�������Ա
		const short* data,                // ������ָ�� 
		const double iso_value,           // ��ֵ���ֵ
		vector< float >& point_position   // ��������ж��㣬x y zֵ�����洢����������Ϊһ��������
	);
	// ����������ʹ�ñ�������
    bool GenerateCollapse(
		const int* sizes,                 // ��ά�����ݴ�С��������Ա��x��y��z 
		const float* spacings,            // ÿ���ȣ�������Ա 
		const int* sample_stride,         // x, y, z���������صĲ�����ȣ�������Ա
		const short* data,                // ������ָ�� 
		const double iso_value,           // ��ֵ���ֵ
		const double decimate_rate,       // ����
		vector< float >& point_position,  // ��������ж��㣬x y zֵ�����洢 
		vector< int >& triangle_index,    // ���������������������������������һ���
        const unsigned int buffer_size    // �򻯹����л���Ĵ�С�����ֵԽ�������Խ�ߣ����ٶ�Խ��
	);

private:
    //static const unsigned int max_buf_size = 30000, min_buf_size = 10000;
};

#endif // #ifndef _ORTHO_SYS_MESH_GENERATOR_H_
