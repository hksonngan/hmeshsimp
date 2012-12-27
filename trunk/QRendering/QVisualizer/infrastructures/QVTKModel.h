/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QVTKModel.h
 * @brief   QVTKModel class definition.
 * 
 * This file defines ...
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/19
 */

#ifndef QVTKMODEL_H
#define QVTKMODEL_H

#define SAMPLE_MAX_SIZE         2048
#define SAMPLE_KERNEL_SIZE      5
#define SAMPLE_KERNEL_SIGMA     0.84089642
#define SAMPLE_KERNEL_PI        3.14159265359

#include <string>
#include <vector>
#include <list>

enum QVTKVersion
{
    VTK_VERSION_1_0     = 0,
    VTK_VERSION_2_0     = 1,
    VTK_VERSION_3_0     = 2,
    VTK_VERSION_UNKNOWN = 3
};

enum QVTKFileFormat
{
    VTK_FILE_FORMAT_ASCII   = 0,
    VTK_FILE_FORMAT_BINARY  = 1,
    VTK_FILE_FORMAT_UNKNOWN = 2
};

enum QVTKDataType
{
    VTK_DATA_TYPE_BIT       = 0,
    VTK_DATA_TYPE_UCHAR     = 1,
    VTK_DATA_TYPE_CHAR      = 2,
    VTK_DATA_TYPE_USHORT    = 3,
    VTK_DATA_TYPE_SHORT     = 4,
    VTK_DATA_TYPE_UINT      = 5,
    VTK_DATA_TYPE_INT       = 6,
    VTK_DATA_TYPE_ULONG     = 7,
    VTK_DATA_TYPE_LONG      = 8,
    VTK_DATA_TYPE_FLOAT     = 9,
    VTK_DATA_TYPE_DOUBLE    = 10,
    VTK_DATA_TYPE_UNKNOWN   = 11
};

enum QVKTDatasetFormat
{
    VTK_DATASET_FORMAT_STRUCTURED_POINTS    = 1,
    VTK_DATASET_FORMAT_STRUCTURED_GRID      = 2,
    VTK_DATASET_FORMAT_RECTILINEAR_GRID     = 3,
    VTK_DATASET_FORMAT_POLYDATA             = 4,
    VTK_DATASET_FORMAT_UNSTRUCTURED_GRID    = 5,
    VTK_DATASET_FORMAT_FIELD                = 6,
    VTK_DATASET_FORMAT_UNKNOWN              = 7
};

enum QVKTDatasetAttribute
{
    VTK_DATASET_ATTRIBUTE_POINT             = 1,
    VTK_DATASET_ATTRIBUTE_CELL              = 2,
    VTK_DATASET_ATTRIBUTE_UNKNOWN           = 3
};

enum QVKTDatasetAttributeType
{
    VTK_DATASET_ATTRIBUTE_TYPE_SCALAR       = 1,
    VTK_DATASET_ATTRIBUTE_TYPE_LOOKUPTABLE  = 2,
    VTK_DATASET_ATTRIBUTE_TYPE_VECTOR       = 3,
    VTK_DATASET_ATTRIBUTE_TYPE_NORMAL       = 4,
    VTK_DATASET_ATTRIBUTE_TYPE_TEXTURE      = 5,
    VTK_DATASET_ATTRIBUTE_TYPE_TENSOR       = 6,
    VTK_DATASET_ATTRIBUTE_TYPE_FIELD        = 7,
    VTK_DATASET_ATTRIBUTE_TYPE_UNKNOWN      = 8
};

enum QVTKCellType
{
    VTK_CELL_TYPE_VERTEX = 1,
    VTK_CELL_TYPE_POLY_VERTEX = 2,
    VTK_CELL_TYPE_LINE = 3,
    VTK_CELL_TYPE_POLY_LINE = 4,
    VTK_CELL_TYPE_TRIANGLE = 5,
    VTK_CELL_TYPE_TRIANGLE_STRIP = 6,
    VTK_CELL_TYPE_POLYGON = 7,
    VTK_CELL_TYPE_PIXEL = 8,
    VTK_CELL_TYPE_QUAD = 9,
    VTK_CELL_TYPE_TETRA = 10,
    VTK_CELL_TYPE_VOXEL = 11,
    VTK_CELL_TYPE_HEXAHEDRON = 12,
    VTK_CELL_TYPE_UNKNOWN = 13
};

class QVTKModel
{
public:
    QVTKModel();
    QVTKModel(const std::string &name);
    ~QVTKModel();
    
    QVTKVersion version;
    std::string header, name;
    QVTKFileFormat fileFormat;
    QVTKDataType dataType;
    QVKTDatasetFormat datasetFormat;

    std::vector<unsigned char> pointData;
    std::vector<unsigned int> cellData;
    std::vector<unsigned int> cellTypes;
    std::list<QVKTDatasetAttribute> datasetAttributes;
    std::list<QVKTDatasetAttributeType> datasetAttributeTypes;
    std::list<QVTKDataType> datasetAttributeDataTypes;
    std::list<std::string> datasetAttributeNames;
    std::list<std::string> datasetAttributeTableNames;
    std::list<std::vector<unsigned char>> datasetAttributeData;
    unsigned int pointNumber, cellNumber;

    unsigned char loadFile(const std::string& name);
    unsigned char saveFile(const std::string& name);
    unsigned char loadBinaryFile(const std::string& name);
    unsigned char saveBinaryFile(const std::string& name);
    unsigned char saveRawFiles(const std::string& name);

    unsigned char readUnstructuredGrid(std::istream &istream, const std::string &content, const int &offset);

    unsigned char initialize();
    unsigned char destroy();

    static std::list<QVTKModel>::iterator find(std::list<QVTKModel> &models, const std::string &name);

private:
    unsigned char getLine(std::istream &istream, std::string &line, int& count);
    std::string getKey(const std::string &line);
    std::string getValue(const std::string &line, const std::string &key);
    QVTKVersion getVersion(const std::string &line);
    QVTKFileFormat getFileFormat(const std::string &line);
    QVTKDataType getDataType(const std::string &line);
    QVKTDatasetFormat getDatasetFormat(const std::string &line);

    unsigned char readData(std::istream &istream, const QVTKDataType &type, const unsigned int &size, std::vector<unsigned char> &data);
    unsigned char readData(const std::string &content, int &count, const QVTKDataType &type, const unsigned int &size, std::vector<unsigned char> &data);
};

#endif  // QVTKMODEL_H