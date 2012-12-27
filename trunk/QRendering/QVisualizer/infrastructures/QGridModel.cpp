/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QGridModel.cpp
 * @brief   QGridModel class declaration.
 * 
 * This file declares the commonly used methods of models defined in QGridModel.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/04/14
 */

#include <gl/glew.h>

#include <iostream>
#include <fstream>

#include "../infrastructures/QSerializer.h"
#include "../utilities/QUtility.h"
#include "QStructure.h"
#include "QGridModel.h"

QGridModel::QGridModel() :
    type(MODEL_UNKNOWN), name(), vertexNumber(0), elementNumber(0), vertexScale(1.0f),
    vertexTranslate(3, 0.0f), vertexBuffer(0), vertexIndexBuffer(0), vbo(0), ebo(0)
{}

QGridModel::QGridModel(const std::string &name) :
    type(MODEL_UNKNOWN), name(name), vertexNumber(0), elementNumber(0), vertexScale(1.0f),
    vertexTranslate(3, 0.0f), vertexBuffer(0), vertexIndexBuffer(0), vbo(0), ebo(0)
{}

QGridModel::~QGridModel()
{
    this->destroy();
}

unsigned char QGridModel::loadBinaryFile(const std::string& name)
{
    std::ifstream file(name.c_str(), std::ios_base::binary);
    if (!file)
    {
        std::cerr << " > ERROR: unable to open input file: \"" << name.c_str() << "\"." <<  std::endl;
        return GL_FALSE;
    }

    unsigned char error(0);
    error |= !QSerializer::read(file, this->name);
    error |= !QSerializerT<QGridModelType>::read(file, this->type);
    error |= !QSerializerT<unsigned int>::read(file, this->vertexNumber);
    error |= !QSerializerT<unsigned int>::read(file, this->elementNumber);
    error |= !QSerializerT<float>::read(file, this->vertexScale);
    error |= !QSerializerT<float>::read(file, this->vertexTranslate);
    error |= !QSerializerT<float>::read(file, this->vertexBuffer);
    error |= !QSerializerT<unsigned int>::read(file, this->vertexIndexBuffer);
    if (error)
    {
        std::cerr << " > ERROR: reading variable failed." <<  std::endl;
        return GL_FALSE;
    }

    file.close();

    return GL_TRUE;
}

unsigned char QGridModel::saveBinaryFile(const std::string& name)
{
    unsigned int edgeNumber(0);
    if (!getEdgeNumber(edgeNumber)) return GL_FALSE;

    std::ofstream file(name.c_str(), std::ios_base::binary);
    if (!file)
    {
        std::cerr << " > ERROR: unable to create output file: \"" << name.c_str() << "\"." <<  std::endl;
        return GL_FALSE;
    }

    unsigned char error(0);
    error |= !QSerializer::write(file, this->name);
    error |= !QSerializerT<QGridModelType>::write(file, this->type);
    error |= !QSerializerT<unsigned int>::write(file, this->vertexNumber);
    error |= !QSerializerT<unsigned int>::write(file, this->elementNumber);
    error |= !QSerializerT<float>::write(file, this->vertexScale);
    error |= !QSerializerT<float>::write(file, this->vertexTranslate);
    error |= !QSerializerT<float>::write(file, this->vertexBuffer);
    error |= !QSerializerT<unsigned int>::write(file, this->vertexIndexBuffer);
    if (error)
    {
        std::cerr << " > ERROR: writing variable failed." <<  std::endl;
        return GL_FALSE;
    }

    file.close();
    
    return GL_TRUE;
}

unsigned char QGridModel::computeVertexNormal(const unsigned char& normalized)
{
    unsigned int edgeNumber(0);
    if (!getEdgeNumber(edgeNumber)) return GL_FALSE;
    
    unsigned int* pIndex = vertexIndexBuffer.data();
    float* pVertex = vertexBuffer.data();
    for (int i = 0; i < elementNumber; i++)
    {
        unsigned int offset0 = pIndex[0] * 6;
        unsigned int offset1 = pIndex[1] * 6;
        unsigned int offset2 = pIndex[2] * 6;

        float* pV0 = pVertex + offset0;
        float* pV1 = pVertex + offset1;
        float* pV2 = pVertex + offset2;
        QVector3 edge0(pV2[0] - pV1[0], pV2[1] - pV1[1], pV2[2] - pV1[2]);
        QVector3 edge1(pV0[0] - pV1[0], pV0[1] - pV1[1], pV0[2] - pV1[2]);

        // Mean Weighted by Angle
        QVector3 cross = QVector3::cross(edge0, edge1);
        float weight = ::asin(QVector3::length(cross) / (QVector3::length(edge0) * QVector3::length(edge1)));
        QVector3 normal = QVector3::normalize(cross) * weight;

        pV0 += 3; pV1 += 3; pV2 += 3;
        pV0[0] += normal.x; pV0[1] += normal.y; pV0[2] += normal.z;
        pV1[0] += normal.x; pV1[1] += normal.y; pV1[2] += normal.z;
        pV2[0] += normal.x; pV2[1] += normal.y; pV2[2] += normal.z;

        pIndex += edgeNumber;
    }

    if (normalized)
    {
        pVertex = vertexBuffer.data() + 3;
        for (int i = 0; i < vertexNumber; i++)
        {
            float scale = 1.0f / sqrt(pVertex[0] * pVertex[0] + pVertex[1] * pVertex[1] + pVertex[2] * pVertex[2]);
            pVertex[0] *= scale;
            pVertex[1] *= scale;
            pVertex[2] *= scale;
            pVertex += 6;
        }
    }
    
    return GL_TRUE;
}

unsigned char QGridModel::computeVertexTranslate()
{
    std::vector<float> maximum(3, -FLT_MAX), minimum(3, FLT_MAX);
    float* pVertex = vertexBuffer.data();
    for (int i = 0; i < vertexNumber; i++)
    {
        float x(pVertex[0]), y(pVertex[1]), z(pVertex[2]);
        if (x > maximum[0]) maximum[0] = x; if (x < minimum[0]) minimum[0] = x;
        if (y > maximum[1]) maximum[1] = y; if (x < minimum[1]) minimum[1] = y;
        if (z > maximum[2]) maximum[2] = z; if (x < minimum[2]) minimum[2] = z;
        pVertex += 6;
    }
    
    vertexScale = 1.0f / sqrt(
        (maximum[0] - minimum[0]) * (maximum[0] - minimum[0]) + 
        (maximum[1] - minimum[1]) * (maximum[1] - minimum[1]) + 
        (maximum[2] - minimum[2]) * (maximum[2] - minimum[2]));

    vertexTranslate[0] = -0.5f * (maximum[0] + minimum[0]);
    vertexTranslate[1] = -0.5f * (maximum[1] + minimum[1]);
    vertexTranslate[2] = -0.5f * (maximum[2] + minimum[2]);

    return GL_TRUE;
}

unsigned char QGridModel::removeRedundantVertex()
{
    if (elementNumber == 0) return GL_TRUE;

    std::vector<unsigned int> indexed(vertexNumber);
    memset(indexed.data(), 0, elementNumber * sizeof(unsigned char));
    for (std::vector<unsigned int>::iterator i = vertexIndexBuffer.begin(); i != vertexIndexBuffer.end(); i++)
        indexed[*i] = 1;

    unsigned int number(0);
    for (int i = 0; i < indexed.size(); i++)
        if (indexed[i] > 0) indexed[i] = ++number;

    for (std::vector<unsigned int>::iterator i = vertexIndexBuffer.begin(); i != vertexIndexBuffer.end(); i++)
        *i = indexed[*i] - 1;

    for (int i = 0; i < indexed.size(); i++)
        if (indexed[i] > 0) indexed[indexed[i] - 1] = i;
    indexed.resize(number);

    vertexNumber = number;
    for (int i = 0; i < vertexNumber; i++)
        if (indexed[i] > i) memcpy(vertexBuffer.data() + i * 6, vertexBuffer.data() + indexed[i] * 6, 6 * sizeof(float));
    vertexBuffer.resize(vertexNumber * 6);

    return GL_TRUE;
}

unsigned char QGridModel::initialize()
{
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertexBuffer.size() * sizeof(float), vertexBuffer.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glBufferData()")) return GL_FALSE;

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, vertexIndexBuffer.size() * sizeof(unsigned int), vertexIndexBuffer.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glBufferData()")) return GL_FALSE;

    return GL_TRUE;
}

unsigned char QGridModel::destroy()
{
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);

    return GL_TRUE;
}

GLuint& QGridModel::getV()
{
    return vbo;
}

GLuint& QGridModel::getE()
{
    return ebo;
}

std::list<QGridModel>::iterator QGridModel::find(std::list<QGridModel> &models, const std::string &name)
{
    for (std::list<QGridModel>::iterator i = models.begin(); i != models.end(); i++)
    {
        if (i->name.find(name) != std::string::npos) return i;
    }
    return models.end();
}

unsigned char QGridModel::getEdgeNumber(unsigned int& number)
{
    switch (type)
    {
    case MODEL_TETRAHEDRON:
        number = 3;
        break;
    case MODEL_HEXAHEDRON:
        number = 4;
        break;
    case MODEL_UNKNOWN:
    case MODEL_HYBRID:
    default:
        std::cerr << " > ERROR: unsupported model type." <<  std::endl;
        return GL_FALSE;
    }
    return GL_TRUE;
}