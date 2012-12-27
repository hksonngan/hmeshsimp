/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @istream    QVTKModel.cpp
 * @brief   QVTKModel class declaration.
 * 
 * This istream declares the commonly used methods of models defined in QVTKModel.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/19
 */

#include <gl/glew.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>

#include "../infrastructures/QSerializer.h"
#include "../utilities/QUtility.h"
#include "../utilities/QTriangle.h"
#include "../utilities/QIO.h"
#include "QStructure.h"
#include "QVTKModel.h"
#include "QModel.h"

// [houtao]
#include "float.h"

QVTKModel::QVTKModel() :
    version(VTK_VERSION_UNKNOWN), header(), name(), fileFormat(VTK_FILE_FORMAT_UNKNOWN), dataType(VTK_DATA_TYPE_UNKNOWN),
    datasetFormat(VTK_DATASET_FORMAT_UNKNOWN), pointNumber(0), pointData(0), cellNumber(0), cellData(0), cellTypes(0),
    datasetAttributes(0), datasetAttributeTypes(0), datasetAttributeDataTypes(0), datasetAttributeNames(0), datasetAttributeTableNames(0), datasetAttributeData(0)
{}

QVTKModel::QVTKModel(const std::string &name) :
    version(VTK_VERSION_UNKNOWN), header(), name(name), fileFormat(VTK_FILE_FORMAT_UNKNOWN), dataType(VTK_DATA_TYPE_UNKNOWN),
    datasetFormat(VTK_DATASET_FORMAT_UNKNOWN), pointNumber(0), pointData(0), cellNumber(0), cellData(0), cellTypes(0),
    datasetAttributes(0), datasetAttributeTypes(0), datasetAttributeDataTypes(0), datasetAttributeNames(0), datasetAttributeTableNames(0), datasetAttributeData(0)
{}

QVTKModel::~QVTKModel()
{
    this->destroy();
}

unsigned char QVTKModel::loadFile(const std::string& name)
{
    std::ifstream istream(name.c_str());
    if (!istream)
    {
        std::cerr << " > ERROR: unable to open input istream: \"" << name.c_str() << "\"." <<  std::endl;
        return GL_FALSE;
    }
    
    std::string content;
    if (!QIO::getFileContent(name, content)) return GL_FALSE;

    std::string line;
    int count(0);
    getLine(istream, line, count);
    version = getVersion(line);
    switch (version)
    {
    case VTK_VERSION_1_0:
    case VTK_VERSION_3_0:
        std::cerr << " > INFO: this version is currently not supported." <<  std::endl;
        return GL_FALSE;
    case VTK_VERSION_2_0:
        break;
    default:
        return GL_FALSE;
    }

    getLine(istream, header, count);

    getLine(istream, line, count);
    fileFormat = getFileFormat(line);
    switch (fileFormat)
    {
    case VTK_FILE_FORMAT_BINARY:
        std::cerr << " > INFO: the format of this istream is currently not supported." <<  std::endl;
        return GL_FALSE;
    case VTK_FILE_FORMAT_ASCII:
        break;
    default:
        return GL_FALSE;
    }

    getLine(istream, line, count);
    datasetFormat = getDatasetFormat(line);
    switch (datasetFormat)
    {
    case VTK_DATASET_FORMAT_STRUCTURED_POINTS:
    case VTK_DATASET_FORMAT_STRUCTURED_GRID:
    case VTK_DATASET_FORMAT_RECTILINEAR_GRID:
    case VTK_DATASET_FORMAT_POLYDATA:
    case VTK_DATASET_FORMAT_FIELD:
        std::cerr << " > INFO: the format of this dataset is currently not supported." <<  std::endl;
        return GL_FALSE;
    case VTK_DATASET_FORMAT_UNSTRUCTURED_GRID:
        return readUnstructuredGrid(istream, content, count);
    default:
        return GL_FALSE;
    }

    return GL_TRUE;
}

unsigned char QVTKModel::loadBinaryFile(const std::string& name)
{
    std::ifstream file(name.c_str(), std::ios_base::binary);
    if (!file)
    {
        std::cerr << " > ERROR: unable to open input file: \"" << name.c_str() << "\"." <<  std::endl;
        return GL_FALSE;
    }

    unsigned char error(0);
    unsigned int size(0);
    error |= !QSerializerT<QVTKVersion>::read(file, this->version);
    error |= !QSerializer::read(file, this->header);
    error |= !QSerializer::read(file, this->name);
    error |= !QSerializerT<QVTKFileFormat>::read(file, this->fileFormat);
    error |= !QSerializerT<QVTKDataType>::read(file, this->dataType);
    error |= !QSerializerT<QVKTDatasetFormat>::read(file, this->datasetFormat);
    error |= !QSerializerT<unsigned int>::read(file, this->pointNumber);
    error |= !QSerializerT<unsigned int>::read(file, this->cellNumber);
    error |= !QSerializerT<unsigned char>::read(file, this->pointData);
    error |= !QSerializerT<unsigned int>::read(file, this->cellData);
    error |= !QSerializerT<unsigned int>::read(file, this->cellTypes);
    error |= !QSerializerT<unsigned int>::read(file, size);
    if (error)
    {
        std::cerr << " > ERROR: writing variable failed." <<  std::endl;
        return GL_FALSE;
    }

    this->datasetAttributes.resize(size);
    this->datasetAttributeTypes.resize(size);
    this->datasetAttributeDataTypes.resize(size);
    this->datasetAttributeNames.resize(size);
    this->datasetAttributeTableNames.resize(size);
    this->datasetAttributeData.resize(size);

    std::list<QVKTDatasetAttribute>::iterator attribute = this->datasetAttributes.begin();
    std::list<QVKTDatasetAttributeType>::iterator attributeType = this->datasetAttributeTypes.begin();
    std::list<QVTKDataType>::iterator attributeDataType = this->datasetAttributeDataTypes.begin();
    std::list<std::string>::iterator attributeName = this->datasetAttributeNames.begin();
    std::list<std::string>::iterator attributeTableName = this->datasetAttributeTableNames.begin();
    std::list<std::vector<unsigned char>>::iterator attributeData = this->datasetAttributeData.begin();
    for (int i = 0; i < size; i++)
    {
        error |= !QSerializerT<QVKTDatasetAttribute>::read(file, *(attribute++));
        error |= !QSerializerT<QVKTDatasetAttributeType>::read(file, *(attributeType++));
        error |= !QSerializerT<QVTKDataType>::read(file, *(attributeDataType++));
        error |= !QSerializer::read(file, *(attributeName++));
        error |= !QSerializer::read(file, *(attributeTableName++));
        error |= !QSerializerT<unsigned char>::read(file, *(attributeData++));
        if (error)
        {
            std::cerr << " > ERROR: writing variable failed." <<  std::endl;
            return GL_FALSE;
        }
    }

    return GL_TRUE;
}

unsigned char QVTKModel::saveBinaryFile(const std::string& name)
{
    std::ofstream file(name.c_str(), std::ios_base::binary);
    if (!file)
    {
        std::cerr << " > ERROR: unable to create output file: \"" << name.c_str() << "\"." <<  std::endl;
        return GL_FALSE;
    }

    unsigned char error(0);
    error |= !QSerializerT<QVTKVersion>::write(file, this->version);
    error |= !QSerializer::write(file, this->header);
    error |= !QSerializer::write(file, this->name);
    error |= !QSerializerT<QVTKFileFormat>::write(file, this->fileFormat);
    error |= !QSerializerT<QVTKDataType>::write(file, this->dataType);
    error |= !QSerializerT<QVKTDatasetFormat>::write(file, this->datasetFormat);
    error |= !QSerializerT<unsigned int>::write(file, this->pointNumber);
    error |= !QSerializerT<unsigned int>::write(file, this->cellNumber);
    error |= !QSerializerT<unsigned char>::write(file, this->pointData);
    error |= !QSerializerT<unsigned int>::write(file, this->cellData);
    error |= !QSerializerT<unsigned int>::write(file, this->cellTypes);
    if (error)
    {
        std::cerr << " > ERROR: writing variable failed." <<  std::endl;
        return GL_FALSE;
    }

    unsigned int size = datasetAttributes.size();
    error |= !QSerializerT<unsigned int>::write(file, size);
    std::list<QVKTDatasetAttribute>::iterator attribute = this->datasetAttributes.begin();
    std::list<QVKTDatasetAttributeType>::iterator attributeType = this->datasetAttributeTypes.begin();
    std::list<QVTKDataType>::iterator attributeDataType = this->datasetAttributeDataTypes.begin();
    std::list<std::string>::iterator attributeName = this->datasetAttributeNames.begin();
    std::list<std::string>::iterator attributeTableName = this->datasetAttributeTableNames.begin();
    std::list<std::vector<unsigned char>>::iterator attributeData = this->datasetAttributeData.begin();
    for (int i = 0; i < size; i++)
    {
        error |= !QSerializerT<QVKTDatasetAttribute>::write(file, *(attribute++));
        error |= !QSerializerT<QVKTDatasetAttributeType>::write(file, *(attributeType++));
        error |= !QSerializerT<QVTKDataType>::write(file, *(attributeDataType++));
        error |= !QSerializer::write(file, *(attributeName++));
        error |= !QSerializer::write(file, *(attributeTableName++));
        error |= !QSerializerT<unsigned char>::write(file, *(attributeData++));
        if (error)
        {
            std::cerr << " > ERROR: writing variable failed." <<  std::endl;
            return GL_FALSE;
        }
    }

    file.close();

    return GL_TRUE;
}

unsigned char QVTKModel::saveRawFiles(const std::string& name)
{
    switch (dataType)
    {
    case VTK_DATA_TYPE_FLOAT:
        break;
    case VTK_DATA_TYPE_BIT:
    case VTK_DATA_TYPE_UCHAR:
    case VTK_DATA_TYPE_CHAR:
    case VTK_DATA_TYPE_USHORT:
    case VTK_DATA_TYPE_SHORT:
    case VTK_DATA_TYPE_UINT:
    case VTK_DATA_TYPE_INT:
    case VTK_DATA_TYPE_ULONG:
    case VTK_DATA_TYPE_LONG:
    case VTK_DATA_TYPE_DOUBLE:
    case VTK_DATA_TYPE_UNKNOWN:
        std::cerr << " > INFO: the data format is currently not supported." <<  std::endl;
        return GL_FALSE;
    }

    float* pPoint = (float*)pointData.data();
    QVector3 positionMin(FLT_MAX, FLT_MAX, FLT_MAX), positionMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int i = 0; i < pointNumber; i++)
    {
        float x = *(pPoint++);
        if (x < positionMin.x) positionMin.x = x;
        if (x > positionMax.x) positionMax.x = x;

        float y = *(pPoint++);
        if (y < positionMin.y) positionMin.y = y;
        if (y > positionMax.y) positionMax.y = y;

        float z = *(pPoint++);
        if (z < positionMin.z) positionMin.z = z;
        if (z > positionMax.z) positionMax.z = z;
    }

    const float a2PS2RI = 1.0f / (std::sqrt(2 * SAMPLE_KERNEL_PI) * SAMPLE_KERNEL_SIGMA);
    const float a2PS2RI3 = a2PS2RI * a2PS2RI * a2PS2RI;
    const float a2S2IN = -1.0f / (2 * SAMPLE_KERNEL_SIGMA * SAMPLE_KERNEL_SIGMA);
    
    QVector3 sampleModelSize = positionMax - positionMin;
    float sampleStepSize = (SAMPLE_MAX_SIZE - 1) / std::max(sampleModelSize.x, std::max(sampleModelSize.y, sampleModelSize.z));
    unsigned int sampleSizeX = (int)(sampleModelSize.x * sampleStepSize + 0.5f) + 1;
    unsigned int sampleSizeY = (int)(sampleModelSize.y * sampleStepSize + 0.5f) + 1;
    unsigned int sampleSizeZ = (int)(sampleModelSize.z * sampleStepSize + 0.5f) + 1;
    unsigned int sampleSize(sampleSizeX * sampleSizeY * sampleSizeZ);
    unsigned int kernelSize(SAMPLE_KERNEL_SIZE * SAMPLE_KERNEL_SIZE * SAMPLE_KERNEL_SIZE);

    std::vector<float> attributeWeights(pointNumber * kernelSize);
    memset(attributeWeights.data(), 0, attributeWeights.size());
    float* pWeight = attributeWeights.data();

    pPoint = (float*)pointData.data();
    for (int i = 0; i < pointNumber; i++)
    {
        float x = (*(pPoint++) - positionMin.x) * sampleStepSize;
        float y = (*(pPoint++) - positionMin.y) * sampleStepSize;
        float z = (*(pPoint++) - positionMin.z) * sampleStepSize;

        float totalWeight(0.0f);
        for (int kz = 0; kz < SAMPLE_KERNEL_SIZE; kz++)
        {
            int iz = (int)(z - SAMPLE_KERNEL_SIZE * 0.5f) + kz + 1;
            if (iz < 0 || iz >= sampleSizeZ) continue;
            for (int ky = 0; ky < SAMPLE_KERNEL_SIZE; ky++)
            {
                int iy = (int)(y - SAMPLE_KERNEL_SIZE * 0.5f) + ky + 1;
                if (iy < 0 || iy >= sampleSizeY) continue;
                for (int kx = 0; kx < SAMPLE_KERNEL_SIZE; kx++)
                {
                    int ix = (int)(x - SAMPLE_KERNEL_SIZE * 0.5f) + kx + 1;
                    if (ix < 0 || ix >= sampleSizeX) continue;
                    float aX2Y2Z2 = (ix - x) * (ix - x) + (iy - y) * (iy - y) + (iz - z) * (iz - z);
                    float weight = a2PS2RI3 * std::exp(aX2Y2Z2 * a2S2IN);
                    pWeight[kx + ky * SAMPLE_KERNEL_SIZE + kz * SAMPLE_KERNEL_SIZE * SAMPLE_KERNEL_SIZE] = weight;
                    totalWeight += weight;
                }
            }
        }

        float weightScale = 1.0f / totalWeight;
        for (int kz = 0; kz < SAMPLE_KERNEL_SIZE; kz++)
            for (int ky = 0; ky < SAMPLE_KERNEL_SIZE; ky++)
                for (int kx = 0; kx < SAMPLE_KERNEL_SIZE; kx++)
                    pWeight[kx + ky * SAMPLE_KERNEL_SIZE + kz * SAMPLE_KERNEL_SIZE * SAMPLE_KERNEL_SIZE] *= weightScale;

        pWeight += kernelSize;
    }

    unsigned int size = datasetAttributes.size();
    std::list<QVKTDatasetAttribute>::iterator attribute = this->datasetAttributes.begin();
    std::list<QVKTDatasetAttributeType>::iterator attributeType = this->datasetAttributeTypes.begin();
    std::list<QVTKDataType>::iterator attributeDataType = this->datasetAttributeDataTypes.begin();
    std::list<std::string>::iterator attributeName = this->datasetAttributeNames.begin();
    std::list<std::string>::iterator attributeTableName = this->datasetAttributeTableNames.begin();
    std::list<std::vector<unsigned char>>::iterator attributeData = this->datasetAttributeData.begin();
    std::vector<float> attributeDataFiltered;
    for (int i = 0; i < size; i++)
    {
        switch (*attribute)
        {
        case VTK_DATASET_ATTRIBUTE_POINT:
            break;
        case VTK_DATASET_ATTRIBUTE_CELL:
        case VTK_DATASET_ATTRIBUTE_UNKNOWN:
            std::cerr << " > INFO: the attribute format is currently not supported." <<  std::endl;
            continue;
        }

        std::string sAttributeType;
        switch (*attributeType)
        {
        case VTK_DATASET_ATTRIBUTE_TYPE_LOOKUPTABLE:
            sAttributeType = "SCALAR";
            break;
        case VTK_DATASET_ATTRIBUTE_TYPE_VECTOR:
        case VTK_DATASET_ATTRIBUTE_TYPE_NORMAL:
            sAttributeType = "VECTOR";
            break;
        case VTK_DATASET_ATTRIBUTE_TYPE_TENSOR:
            sAttributeType = "TENSOR";
            break;
        case VTK_DATASET_ATTRIBUTE_TYPE_SCALAR:
        case VTK_DATASET_ATTRIBUTE_TYPE_TEXTURE:
        case VTK_DATASET_ATTRIBUTE_TYPE_FIELD:
        case VTK_DATASET_ATTRIBUTE_TYPE_UNKNOWN:
            std::cerr << " > INFO: the attribute type is currently not supported." <<  std::endl;
            continue;
        }

        std::string sAttributeDataType;
        unsigned int attributeSize(0);
        switch (*attributeDataType)
        {
        case VTK_DATA_TYPE_FLOAT:
            sAttributeDataType = "FLOAT";
            attributeSize = sizeof(float);
            break;
        case VTK_DATA_TYPE_BIT:
        case VTK_DATA_TYPE_UCHAR:
        case VTK_DATA_TYPE_CHAR:
        case VTK_DATA_TYPE_USHORT:
        case VTK_DATA_TYPE_SHORT:
        case VTK_DATA_TYPE_UINT:
        case VTK_DATA_TYPE_INT:
        case VTK_DATA_TYPE_ULONG:
        case VTK_DATA_TYPE_LONG:
        case VTK_DATA_TYPE_DOUBLE:
        case VTK_DATA_TYPE_UNKNOWN:
            std::cerr << " > INFO: the attribute data type is currently not supported." <<  std::endl;
            continue;
        }

        int position = name.find_last_of("/\\") + 1;
        std::string filePath = name.substr(0, position);
        std::string fileName = name.substr(position) + "_" + *attributeName;
        std::string objFileName = fileName + ".raw";
        std::string datFileName = fileName + ".dat";
        std::ofstream datFileStream((filePath + datFileName).c_str());
        if (!datFileStream)
        {
            std::cerr << " > ERROR: unable to create output file: \"" << filePath.c_str() << datFileName.c_str() << "\"." <<  std::endl;
            return GL_FALSE;
        }

        unsigned int attributeNumber(attributeData->size() / (pointNumber * attributeSize));
        datFileStream << "ObjectFileName: " << objFileName.c_str() << std::endl;
        datFileStream << "Resolution: "     << sampleSizeX << " " << sampleSizeY << " " << sampleSizeZ << std::endl;
        datFileStream << "SliceThickness: " << sampleStepSize << " " << sampleStepSize << " " << sampleStepSize << std::endl;
        datFileStream << "Dimension: "      << attributeNumber << std::endl;
        datFileStream << "Type: "           << sAttributeType.c_str() << std::endl;
        datFileStream << "Format: "         << sAttributeDataType.c_str() << std::endl;

        datFileStream.close();

        std::ofstream objFileStream((filePath + objFileName).c_str(), std::ios_base::binary);
        if (!objFileStream)
        {
            std::cerr << " > ERROR: unable to create output file: \"" << filePath.c_str() << objFileName.c_str() << "\"." <<  std::endl;
            return GL_FALSE;
        }

        attributeDataFiltered.resize(sampleSize * attributeNumber);
        memset(attributeDataFiltered.data(), 0, attributeDataFiltered.size() * sizeof(float));
        
        float* pAttributeData = (float*)attributeData->data();
        pPoint = (float*)pointData.data();
        pWeight = attributeWeights.data();
        for (int i = 0; i < pointNumber; i++)
        {
            float x = (*(pPoint++) - positionMin.x) * sampleStepSize;
            float y = (*(pPoint++) - positionMin.y) * sampleStepSize;
            float z = (*(pPoint++) - positionMin.z) * sampleStepSize;

            for (int kz = 0; kz < SAMPLE_KERNEL_SIZE; kz++)
            {
                int iz = (int)(z - SAMPLE_KERNEL_SIZE * 0.5f) + kz + 1;
                if (iz < 0 || iz >= sampleSizeZ) continue;
                for (int ky = 0; ky < SAMPLE_KERNEL_SIZE; ky++)
                {
                    int iy = (int)(y - SAMPLE_KERNEL_SIZE * 0.5f) + ky + 1;
                    if (iy < 0 || iy >= sampleSizeY) continue;
                    for (int kx = 0; kx < SAMPLE_KERNEL_SIZE; kx++)
                    {
                        int ix = (int)(x - SAMPLE_KERNEL_SIZE * 0.5f) + kx + 1;
                        if (ix < 0 || ix >= sampleSizeX) continue;

                        float weight = pWeight[kx + ky * SAMPLE_KERNEL_SIZE + kz * SAMPLE_KERNEL_SIZE * SAMPLE_KERNEL_SIZE];
                        float* pAttributeDataFiltered = attributeDataFiltered.data() + ix + iy * sampleSizeX + iz * sampleSizeY * sampleSizeX;
                        for (int j = 0; j < attributeNumber; j++)
                            pAttributeDataFiltered[j] += pAttributeData[j] * weight;
                    }
                }
            }

            pAttributeData += attributeNumber;
            pWeight += kernelSize;
        }

        objFileStream.write((char*)attributeDataFiltered.data(), attributeDataFiltered.size() * sizeof(float));
        objFileStream.close();

        attribute++;
        attributeType++;
        attributeDataType++;
        attributeName++;
        attributeTableName++;
        attributeData++;
    }

    return GL_TRUE;
}

unsigned char QVTKModel::readUnstructuredGrid(std::istream &istream, const std::string &content, const int &offset)
{
    std::string line;

    // POINTS

    int count(offset);
    getLine(istream, line, count);
    std::string value = getValue(line, "POINTS ");
    if (value.empty()) return GL_FALSE;

    int number(0);
    std::stringstream stream;
    stream << value;
    stream >> number;
    if (number <= 0)
    {
        std::cerr << " > ERROR: illegal POINTS number." <<  std::endl;
        return GL_FALSE;
    }
    pointNumber = number;

    stream >> value;
    dataType = getDataType(value);
    // if (!readData(istream, dataType, pointNumber * 3, pointData)) return GL_FALSE;
    if (!readData(content, count, dataType, pointNumber * 3, pointData)) return GL_FALSE;
    istream.seekg(count, std::ios::beg);
    
    // CELLS

    getLine(istream, line, count);
    value = getValue(line, "CELLS ");
    if (value.empty()) return GL_FALSE;

    stream.clear();
    stream.str(value);
    stream >> number;
    if (number <= 0)
    {
        std::cerr << " > ERROR: illegal CELLS number." <<  std::endl;
        return GL_FALSE;
    }
    cellNumber = number;

    stream >> number;
    if (number <= 0)
    {
        std::cerr << " > ERROR: illegal CELLS size." <<  std::endl;
        return GL_FALSE;
    }
    unsigned int cellSize(number);

    cellData.resize(cellSize);
    // QUtilityTemplate<unsigned int>::read(istream, cellData.data(), cellSize);
    QUtilityTemplate<unsigned int>::read(content, count, cellData.data(), cellSize);
    istream.seekg(count, std::ios::beg);

    // CELL_TYPES

    getLine(istream, line, count);
    value = getValue(line, "CELL_TYPES ");
    if (value.empty()) return GL_FALSE;

    stream.clear();
    stream.str(value);
    stream >> number;
    if (number != cellNumber)
    {
        std::cerr << " > ERROR: illegal CELL_TYPES number." <<  std::endl;
        return GL_FALSE;
    }

    cellTypes.resize(cellNumber);
    // QUtilityTemplate<unsigned int>::read(istream, cellTypes.data(), cellNumber);
    QUtilityTemplate<unsigned int>::read(content, count, cellTypes.data(), cellNumber);
    istream.seekg(count, std::ios::beg);

    getLine(istream, line, count);
    while (!istream.eof())
    {
        QVKTDatasetAttribute attribute(VTK_DATASET_ATTRIBUTE_UNKNOWN);
        std::string key = getKey(line);
        if (key.compare("POINT_DATA") == 0)
        {
            attribute = VTK_DATASET_ATTRIBUTE_POINT;
        }
        else if (key.compare("CELL_DATA") == 0)
        {
            attribute = VTK_DATASET_ATTRIBUTE_CELL;
        }
        else
        {
            std::cerr << " > ERROR: illegal dataset attributes." <<  std::endl;
            return GL_FALSE;
        }

        value = getValue(line, key);
        if (value.empty()) return GL_FALSE;

        stream.clear();
        stream.str(value);
        stream >> number;
        if ((attribute == VTK_DATASET_ATTRIBUTE_POINT && number != pointNumber) || (attribute == VTK_DATASET_ATTRIBUTE_CELL && number != cellNumber))
        {
            std::cerr << " > ERROR: illegal POINT_DATA number." <<  std::endl;
            return GL_FALSE;
        }
        unsigned int attributeSize(number);

        std::vector<unsigned char> attributeData(0);
        QVKTDatasetAttributeType attributeType(VTK_DATASET_ATTRIBUTE_TYPE_UNKNOWN);
        QVTKDataType attributeDataType(VTK_DATA_TYPE_UNKNOWN);
        std::string attributeName, attributeTableName;
        unsigned int attributeNumber(0);
        getLine(istream, line, count);
        while (!istream.eof())
        {
            attributeTableName.assign("default");

            std::string key = getKey(line);
            if (key.compare("SCALARS") == 0)
            {
                attributeType = VTK_DATASET_ATTRIBUTE_TYPE_SCALAR;
            }
            else if (key.compare("LOOKUP_TABLE") == 0)
            {
                attributeType = VTK_DATASET_ATTRIBUTE_TYPE_LOOKUPTABLE;
            }
            else if (key.compare("VECTORS") == 0)
            {
                attributeType = VTK_DATASET_ATTRIBUTE_TYPE_VECTOR;
                attributeNumber = 3;
            }
            else if (key.compare("NORMALS") == 0)
            {
                attributeType = VTK_DATASET_ATTRIBUTE_TYPE_NORMAL;
                attributeNumber = 3;
            }
            else if (key.compare("TEXTURE_COORDINATES") == 0)
            {
                attributeType = VTK_DATASET_ATTRIBUTE_TYPE_TEXTURE;
            }
            else if (key.compare("TENSORS") == 0)
            {
                attributeType = VTK_DATASET_ATTRIBUTE_TYPE_TENSOR;
                attributeNumber = 9;
            }
            else if (key.compare("FIELD") == 0)
            {
                attributeType = VTK_DATASET_ATTRIBUTE_TYPE_FIELD;
            }
            else
            {
                std::cerr << " > ERROR: illegal dataset attributes." <<  std::endl;
                return GL_FALSE;
            }

            value = getValue(line, key);
            if (value.empty()) return GL_FALSE;

            stream.clear();
            stream.str(value);

            if (attributeType == VTK_DATASET_ATTRIBUTE_TYPE_LOOKUPTABLE)
            {
                stream >> attributeTableName;
            }
            else
            {
                stream >> attributeName;
            }

            if (attributeType == VTK_DATASET_ATTRIBUTE_TYPE_LOOKUPTABLE)
            {
                unsigned int size(0);
                stream >> size;
                if (size > 0)
                {
                    attributeSize = size;
                    attributeNumber = 4;
                }
            }
            else if (attributeType == VTK_DATASET_ATTRIBUTE_TYPE_TEXTURE || attributeType == VTK_DATASET_ATTRIBUTE_TYPE_FIELD)
            {
                stream >> attributeNumber;
            }
            else
            {
                stream >> value;
                attributeDataType = getDataType(value);
            }

            if (attributeType == VTK_DATASET_ATTRIBUTE_TYPE_SCALAR)
            {
                unsigned int number(0);
                stream >> number;
                attributeNumber = number > 0 ? number : 1;
            }
            else
            {
                if (!readData(content, count, attributeDataType, attributeSize * attributeNumber, attributeData)) return GL_FALSE;
                istream.seekg(count, std::ios::beg);

                datasetAttributes.push_front(attribute);
                datasetAttributeTypes.push_front(attributeType);
                datasetAttributeDataTypes.push_front(attributeDataType);
                datasetAttributeNames.push_front(attributeName);
                datasetAttributeTableNames.push_front(attributeTableName);
                datasetAttributeData.push_front(attributeData);
            }

            getLine(istream, line, count);
        }
    }
    
    return GL_TRUE;
}

unsigned char QVTKModel::saveFile(const std::string& name)
{
    QModel model(this->name);
    model.type = MODEL_TRIANGLE;
    model.vertexNumber = this->pointNumber;
    model.vertexBuffer.resize(model.vertexNumber * 6);
    switch (dataType)
    {
    case VTK_DATA_TYPE_BIT:
        std::cerr << " > INFO: the data format is currently not supported." <<  std::endl;
        return GL_FALSE;
    case VTK_DATA_TYPE_UCHAR:
        QUtilityTemplate<unsigned char>::convert(pointData.data(), model.vertexBuffer.data(), model.vertexNumber, 6);
        break;
    case VTK_DATA_TYPE_CHAR:
        QUtilityTemplate<char>::convert(pointData.data(), model.vertexBuffer.data(), model.vertexNumber, 6);
        break;
    case VTK_DATA_TYPE_USHORT:
        QUtilityTemplate<unsigned short>::convert(pointData.data(), model.vertexBuffer.data(), model.vertexNumber, 6);
        break;
    case VTK_DATA_TYPE_SHORT:
        QUtilityTemplate<short>::convert(pointData.data(), model.vertexBuffer.data(), model.vertexNumber, 6);
        break;
    case VTK_DATA_TYPE_UINT:
        QUtilityTemplate<unsigned int>::convert(pointData.data(), model.vertexBuffer.data(), model.vertexNumber, 6);
        break;
    case VTK_DATA_TYPE_INT:
        QUtilityTemplate<int>::convert(pointData.data(), model.vertexBuffer.data(), model.vertexNumber, 6);
        break;
    case VTK_DATA_TYPE_ULONG:
        QUtilityTemplate<unsigned long>::convert(pointData.data(), model.vertexBuffer.data(), model.vertexNumber, 6);
        break;
    case VTK_DATA_TYPE_LONG:
        QUtilityTemplate<long>::convert(pointData.data(), model.vertexBuffer.data(), model.vertexNumber, 6);
        break;
    case VTK_DATA_TYPE_DOUBLE:
        QUtilityTemplate<double>::convert(pointData.data(), model.vertexBuffer.data(), model.vertexNumber, 6);
        break;
    case VTK_DATA_TYPE_FLOAT:
        QUtilityTemplate<float>::convert(pointData.data(), model.vertexBuffer.data(), model.vertexNumber, 6);
        break;
    default:
        return GL_FALSE;
    }
    
    model.elementNumber = 0;
    int offset(0), maxNumber(0);
    for (std::vector<unsigned int>::iterator i = cellTypes.begin(); i != cellTypes.end(); i++)
    {
        int size(cellData[offset]);
        offset += size + 1;
        int numberOfPoints(0);
        switch (*i)
        {
        case VTK_CELL_TYPE_POLYGON:
            numberOfPoints = size - 1;
            if (numberOfPoints > maxNumber) maxNumber = numberOfPoints;
            model.elementNumber += numberOfPoints - 2;
        	break;
        case VTK_CELL_TYPE_QUAD:
            numberOfPoints = size;
            if (numberOfPoints > maxNumber) maxNumber = numberOfPoints;
            model.elementNumber += numberOfPoints - 2;
            break;
        default:
            break;
        }
    }
    model.vertexIndexBuffer.resize(model.elementNumber * 3);

    std::vector<float> pointList(maxNumber * 2);
    std::vector<int> segmentList(maxNumber * 2);
    for (int i = 0; i < maxNumber; i++)
    {
        segmentList[i * 2 + 0] = i;
        segmentList[i * 2 + 1] = i + 1;
    }

    struct triangulateio in, out;
    memset(&in, 0, sizeof(triangulateio));
    memset(&out, 0, sizeof(triangulateio));
    in.pointlist = pointList.data();
    in.segmentlist = segmentList.data();

    offset = 0;
    int* pVertexIndex = (int*)model.vertexIndexBuffer.data();
    for (std::vector<unsigned int>::iterator i = cellTypes.begin(); i != cellTypes.end(); i++)
    {
        int size(cellData[offset]), numberOfPoints(0);
        switch (*i)
        {
        case VTK_CELL_TYPE_POLYGON:
            numberOfPoints = size - 1;
            break;
        case VTK_CELL_TYPE_QUAD:
            numberOfPoints = size;
            break;
        default:
            break;
        }
        if (numberOfPoints == 0) continue;

        // pointList.resize(number * 2);
        unsigned int* pCell = cellData.data() + offset + 1;
        for (int j = 0; j < numberOfPoints; j++)
        {
            float* pCoordinate = model.vertexBuffer.data() + pCell[j] * 6;
            pointList.at(j * 2 + 0) = pCoordinate[0] + pCoordinate[1] + pCoordinate[2];
            pointList.at(j * 2 + 1) = pCoordinate[0] - pCoordinate[1];
        }

        segmentList.at(numberOfPoints * 2 - 1) = 0;
        in.numberofpoints = in.numberofsegments = numberOfPoints;
        triangulate("pzQNBP", &in, &out, NULL);
        segmentList.at(numberOfPoints * 2 - 1) = numberOfPoints;
        
        int numberOfTriangles = std::min(numberOfPoints - 2, out.numberoftriangles);
        int* pTriangle = out.trianglelist;
        for (int j = 0; j < numberOfTriangles; j++)
        {
            unsigned char sorted(0);
            int s = pTriangle[0] + pTriangle[1] + pTriangle[2];
            for (int m = 0; m < 2 && !sorted; m++)
                for (int n = m + 1; n < 3 && !sorted; n++)
                {
                    int x(pTriangle[m]), y(pTriangle[n]);
                    if (x - y == 1 || numberOfPoints + x - y == 1)
                    {
                        pVertexIndex[j * 3 + 0] = pCell[x]; pVertexIndex[j * 3 + 1] = pCell[y]; pVertexIndex[j * 3 + 2] = pCell[s - x - y];
                        sorted = 1;
                    }
                    if (y - x == 1 || numberOfPoints + y - x == 1)
                    {
                        pVertexIndex[j * 3 + 0] = pCell[y]; pVertexIndex[j * 3 + 1] = pCell[x]; pVertexIndex[j * 3 + 2] = pCell[s - x - y];
                        sorted = 1;
                    }
                }
            pTriangle += 3;
        }

        free(out.trianglelist);
        out.trianglelist= NULL;

        offset += size + 1;
        pVertexIndex += (numberOfPoints - 2) * 3;
    }
    
    model.removeRedundantVertex();
    model.computeVertexTranslate();
    model.computeVertexNormal();
    model.saveBinaryFile(name);

    return GL_TRUE;
}

unsigned char QVTKModel::initialize()
{
    return GL_TRUE;
}

unsigned char QVTKModel::destroy()
{
    return GL_TRUE;
}

std::list<QVTKModel>::iterator QVTKModel::find(std::list<QVTKModel> &models, const std::string &name)
{
    for (std::list<QVTKModel>::iterator i = models.begin(); i != models.end(); i++)
    {
        if (i->name.find(name) != std::string::npos) return i;
    }
    return models.end();
}

QVTKVersion QVTKModel::getVersion(const std::string &line)
{
    std::string value = getValue(line, "# vtk DataFile Version ");
    if (value.empty()) return VTK_VERSION_UNKNOWN;

    std::stringstream stream;
    float version(0.0f);
    stream << value;
    stream >> version;

    switch ((int)version)
    {
    case 1:
        return VTK_VERSION_1_0;
    case 2:
        return VTK_VERSION_2_0;
    case 3:
        return VTK_VERSION_3_0;
    default:
        std::cerr << " > ERROR: unsupported version." <<  std::endl;
        return VTK_VERSION_UNKNOWN;
    }
}

QVTKFileFormat QVTKModel::getFileFormat(const std::string &line)
{
    if (line.compare("ASCII") == 0)
    {
        return VTK_FILE_FORMAT_ASCII;
    }
    else if (line.compare("BINARY") == 0)
    {
        return VTK_FILE_FORMAT_BINARY;
    }
    else
    {
        std::cerr << " > ERROR: unsupported istream format." <<  std::endl;
        return VTK_FILE_FORMAT_UNKNOWN;
    }
}

QVKTDatasetFormat QVTKModel::getDatasetFormat(const std::string &line)
{
    std::string value = getValue(line, "DATASET ");
    if (value.empty()) return VTK_DATASET_FORMAT_UNKNOWN;
    
    if (value.compare("STRUCTURED_POINTS") == 0)
    {
        return VTK_DATASET_FORMAT_STRUCTURED_POINTS;
    }
    else if (value.compare("STRUCTURED_GRID") == 0)
    {
        return VTK_DATASET_FORMAT_STRUCTURED_GRID;
    }
    else if (value.compare("RECTILINEAR_GRID") == 0)
    {
        return VTK_DATASET_FORMAT_RECTILINEAR_GRID;
    }
    else if (value.compare("FORMAT_POLYDATA") == 0)
    {
        return VTK_DATASET_FORMAT_POLYDATA;
    }
    else if (value.compare("UNSTRUCTURED_GRID") == 0)
    {
        return VTK_DATASET_FORMAT_UNSTRUCTURED_GRID;
    }
    else if (value.compare("FIELD") == 0)
    {
        return VTK_DATASET_FORMAT_FIELD;
    }
    else
    {
        std::cerr << " > ERROR: unsupported dataset format." <<  std::endl;
        return VTK_DATASET_FORMAT_UNKNOWN;
    }
}

QVTKDataType QVTKModel::getDataType(const std::string &value)
{
    if (value.compare("bit") == 0)
    {
        return VTK_DATA_TYPE_BIT;
    }
    else if (value.compare("unsigned_char") == 0)
    {
        return VTK_DATA_TYPE_UCHAR;
    }
    else if (value.compare("char") == 0)
    {
        return VTK_DATA_TYPE_CHAR;
    }
    else if (value.compare("unsigned_short") == 0)
    {
        return VTK_DATA_TYPE_USHORT;
    }
    else if (value.compare("short") == 0)
    {
        return VTK_DATA_TYPE_SHORT;
    }
    else if (value.compare("unsigned_int") == 0)
    {
        return VTK_DATA_TYPE_UINT;
    }
    else if (value.compare("int") == 0)
    {
        return VTK_DATA_TYPE_INT;
    }
    else if (value.compare("unsigned_long") == 0)
    {
        return VTK_DATA_TYPE_ULONG;
    }
    else if (value.compare("long") == 0)
    {
        return VTK_DATA_TYPE_LONG;
    }
    else if (value.compare("float") == 0)
    {
        return VTK_DATA_TYPE_FLOAT;
    }
    else if (value.compare("double") == 0)
    {
        return VTK_DATA_TYPE_DOUBLE;
    }
    else
    {
        std::cerr << " > ERROR: unsupported data type." <<  std::endl;
        return VTK_DATA_TYPE_UNKNOWN;
    }
}

std::string QVTKModel::getKey(const std::string &line)
{
    int start(0);
    if ((start = line.find_first_not_of(' ')) == std::string::npos)
    {
        std::cerr << " > ERROR: illegal dataset description." <<  std::endl;
        return "";
    }
    int end(0);
    if ((end = line.find_first_of(' ', start)) == std::string::npos)
        return line.substr(start);
    else
        return line.substr(start, end - start);
}

std::string QVTKModel::getValue(const std::string &line, const std::string &key)
{
    int start(0);
    if ((start = line.find_first_not_of(' ')) == std::string::npos)
    {
        std::cerr << " > ERROR: illegal " << key.c_str() << " description." <<  std::endl;
        return "";
    }
    if (line.find(key.c_str(), start) != 0)
    {
        std::cerr << " > ERROR: illegal " << key.c_str() << " description." <<  std::endl;
        return "";
    }
    return line.substr(key.size());
}

unsigned char QVTKModel::getLine(std::istream &istream, std::string& line, int& count)
{
    std::string buffer(1024, 0);
    while (!istream.eof())
    {
        istream.getline((char*)buffer.data(), buffer.size());
        line.assign(buffer.data());
        count += istream.gcount();

        //getline(istream, line);
        int end(0);
        if ((end = line.find_last_not_of(' ')) != std::string::npos)
        {
            line = line.substr(0, end + 1);
            return GL_TRUE;
        }
    }
    return GL_FALSE;
}

unsigned char QVTKModel::readData(std::istream &istream, const QVTKDataType &type, const unsigned int &size, std::vector<unsigned char> &data)
{
    switch (type)
    {
    case VTK_DATA_TYPE_BIT:
        std::cerr << " > INFO: the data format is currently not supported." <<  std::endl;
        return GL_FALSE;
    case VTK_DATA_TYPE_UCHAR:
        data.resize(size * sizeof(unsigned char));
        QUtilityTemplate<unsigned char>::read(istream, data.data(), size);
        break;
    case VTK_DATA_TYPE_CHAR:
        data.resize(size * sizeof(char));
        QUtilityTemplate<char>::read(istream, data.data(), size);
        break;
    case VTK_DATA_TYPE_USHORT:
        data.resize(size * sizeof(unsigned short));
        QUtilityTemplate<unsigned short>::read(istream, data.data(), size);
        break;
    case VTK_DATA_TYPE_SHORT:
        data.resize(size * sizeof(short));
        QUtilityTemplate<short>::read(istream, data.data(), size);
        break;
    case VTK_DATA_TYPE_UINT:
        data.resize(size * sizeof(unsigned int));
        QUtilityTemplate<unsigned int>::read(istream, data.data(), size);
        break;
    case VTK_DATA_TYPE_INT:
        data.resize(size * sizeof(int));
        QUtilityTemplate<int>::read(istream, data.data(), size);
        break;
    case VTK_DATA_TYPE_ULONG:
        data.resize(size * sizeof(unsigned long));
        QUtilityTemplate<unsigned long>::read(istream, data.data(), size);
        break;
    case VTK_DATA_TYPE_LONG:
        data.resize(size * sizeof(long));
        QUtilityTemplate<long>::read(istream, data.data(), size);
        break;
    case VTK_DATA_TYPE_DOUBLE:
        data.resize(size * sizeof(double));
        QUtilityTemplate<double>::read(istream, data.data(), size);
        break;
    case VTK_DATA_TYPE_FLOAT:
        data.resize(size * sizeof(float));
        QUtilityTemplate<float>::read(istream, data.data(), size);
        break;
    default:
        return GL_FALSE;
    }
    return GL_TRUE;
}

unsigned char QVTKModel::readData(const std::string &content, int &count, const QVTKDataType &type, const unsigned int &size, std::vector<unsigned char> &data)
{
    switch (type)
    {
    case VTK_DATA_TYPE_BIT:
        std::cerr << " > INFO: the data format is currently not supported." <<  std::endl;
        return GL_FALSE;
    case VTK_DATA_TYPE_UCHAR:
        data.resize(size * sizeof(unsigned char));
        QUtilityTemplate<unsigned char>::read(content, count, data.data(), size);
        break;
    case VTK_DATA_TYPE_CHAR:
        data.resize(size * sizeof(char));
        QUtilityTemplate<char>::read(content, count, data.data(), size);
        break;
    case VTK_DATA_TYPE_USHORT:
        data.resize(size * sizeof(unsigned short));
        QUtilityTemplate<unsigned short>::read(content, count, data.data(), size);
        break;
    case VTK_DATA_TYPE_SHORT:
        data.resize(size * sizeof(short));
        QUtilityTemplate<short>::read(content, count, data.data(), size);
        break;
    case VTK_DATA_TYPE_UINT:
        data.resize(size * sizeof(unsigned int));
        QUtilityTemplate<unsigned int>::read(content, count, data.data(), size);
        break;
    case VTK_DATA_TYPE_INT:
        data.resize(size * sizeof(int));
        QUtilityTemplate<int>::read(content, count, data.data(), size);
        break;
    case VTK_DATA_TYPE_ULONG:
        data.resize(size * sizeof(unsigned long));
        QUtilityTemplate<unsigned long>::read(content, count, data.data(), size);
        break;
    case VTK_DATA_TYPE_LONG:
        data.resize(size * sizeof(long));
        QUtilityTemplate<long>::read(content, count, data.data(), size);
        break;
    case VTK_DATA_TYPE_DOUBLE:
        data.resize(size * sizeof(double));
        QUtilityTemplate<double>::read(content, count, data.data(), size);
        break;
    case VTK_DATA_TYPE_FLOAT:
        data.resize(size * sizeof(float));
        QUtilityTemplate<float>::read(content, count, data.data(), size);
        break;
    default:
        return GL_FALSE;
    }
    return GL_TRUE;
}