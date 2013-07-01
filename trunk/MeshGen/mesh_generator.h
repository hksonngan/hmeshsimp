///////////////////////////////////////////////////////////
//  mesh_generator.h
//  Implementation of the Class MeshGenerator
//  Created on:      05-三月-2013 13:31:57
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
	
	// 网格生成
	bool GenerateMesh(
		const int* sizes,                 // 三维体数据大小，三个成员，x，y，z 
		const float* spacings,            // 每层厚度，三个成员 
		const int* sample_stride,         // x, y, z方向上的采样宽度，三个成员
		const short* data,                // 体数据指针 
		const double iso_value,           // 等值面的值
		vector< float >& point_position   // 输出网格中顶点，x y z值连续存储，三个顶点为一个三角形
	);
	// 网格生成中使用边收缩简化
    bool GenerateCollapse(
		const int* sizes,                 // 三维体数据大小，三个成员，x，y，z 
		const float* spacings,            // 每层厚度，三个成员 
		const int* sample_stride,         // x, y, z方向上体素的采样宽度，三个成员
		const short* data,                // 体数据指针 
		const double iso_value,           // 等值面的值
		const double decimate_rate,       // 简化率
		vector< float >& point_position,  // 输出网格中顶点，x y z值连续存储 
		vector< int >& triangle_index,    // 输出网格中三角形索引，三个索引连在一块儿
        const unsigned int buffer_size    // 简化过程中缓存的大小，这个值越大简化质量越高，但速度越慢
	);

private:
    //static const unsigned int max_buf_size = 30000, min_buf_size = 10000;
};

#endif // #ifndef _ORTHO_SYS_MESH_GENERATOR_H_
