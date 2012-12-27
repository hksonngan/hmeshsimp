/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QModel.cpp
 * @brief   QModel class declaration.
 * 
 * This file declares the commonly used methods of models defined in QModel.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/19
 */

#include <gl/glew.h>

#include <iostream>
#include <fstream>

#include "../infrastructures/QSerializer.h"
#include "../utilities/QUtility.h"
#include "QStructure.h"
#include "QModel.h"

// [houtao]
#include "float.h"
#include "math.h"

QModel::QModel() :
    type(MODEL_UNKNOWN), name(), vertexNumber(0), elementNumber(0), vertexScale(1.0f),
    vertexTranslate(3, 0.0f), vertexBuffer(0), vertexIndexBuffer(0), vbo(0), ebo(0)
{}

QModel::QModel(const std::string &name) :
    type(MODEL_UNKNOWN), name(name), vertexNumber(0), elementNumber(0), vertexScale(1.0f),
    vertexTranslate(3, 0.0f), vertexBuffer(0), vertexIndexBuffer(0), vbo(0), ebo(0)
{}

QModel::~QModel()
{
    this->destroy();
}

unsigned char QModel::loadGemFile(const std::string& name)
{
    std::ifstream file(name.c_str());
    if (!file)
    {
        std::cerr << " > ERROR: unable to open input file: \"" << name.c_str() << "\"." <<  std::endl;
        return GL_FALSE;
    }
    
    std::string line;
    std::vector<unsigned int> size(3, 0);
    unsigned int edgeNumber(0);
    file >> vertexNumber >> edgeNumber >> size[0] >> size[1] >> size[2];
    getline(file, line);

    type = MODEL_QUAD;

    if (vertexNumber != (size[0] + 1) * (size[1] + 1) * (size[2] + 1))
    {
        std::cerr << " > ERROR: illegal arguments." <<  std::endl;
        return GL_FALSE;
    }

    unsigned int vertexBufferSize = vertexNumber * 6;
    vertexBuffer.resize(vertexBufferSize);
    if (vertexBuffer.size() != vertexBufferSize)
    {
        std::cerr << " > ERROR: not enough memory." <<  std::endl;
        return GL_FALSE;
    }

    float* pVertex = vertexBuffer.data();
    std::vector<float> maximum(3, -FLT_MAX), minimum(3, FLT_MAX);
    for (int i = 0; i <= size[2]; i++)
        for (int j = 0; j <= size[0]; j++)
            for (int k = 0; k <= size[1]; k++)
            {
                file >> *(pVertex++);
                file >> *(pVertex++);
                file >> *(pVertex++);
                pVertex += 3;
            }

    computeVertexTranslate();
    
    elementNumber = (size[0] * size[1] + size[2] * size[1] + size[2] * size[0]) * 2;
    unsigned int vertexIndexBufferSize = elementNumber * 4;
    vertexIndexBuffer.resize(vertexIndexBufferSize);
    if (vertexIndexBuffer.size() != vertexIndexBufferSize)
    {
        std::cerr << " > ERROR: not enough memory." <<  std::endl;
        return GL_FALSE;
    }
    
    unsigned int offset[3] = { 1, size[1] + 1, (size[0] + 1) * (size[1] + 1) };
    unsigned int* pIndex = vertexIndexBuffer.data();
    for (int j = 0; j < size[0]; j++)
        for (int k = 0; k < size[1]; k++)
        {
            *(pIndex++) = (k + 1) * offset[0] + (j + 0) * offset[1];
            *(pIndex++) = (k + 0) * offset[0] + (j + 0) * offset[1];
            *(pIndex++) = (k + 0) * offset[0] + (j + 1) * offset[1];
            *(pIndex++) = (k + 1) * offset[0] + (j + 1) * offset[1];

            *(pIndex++) = (k + 1) * offset[0] + (j + 1) * offset[1] + size[2] * offset[2];
            *(pIndex++) = (k + 0) * offset[0] + (j + 1) * offset[1] + size[2] * offset[2];
            *(pIndex++) = (k + 0) * offset[0] + (j + 0) * offset[1] + size[2] * offset[2];
            *(pIndex++) = (k + 1) * offset[0] + (j + 0) * offset[1] + size[2] * offset[2];
        }

    for (int i = 0; i < size[2]; i++)
        for (int k = 0; k < size[1]; k++)
        {
            *(pIndex++) = (k + 0) * offset[0] + (i + 0) * offset[2];
            *(pIndex++) = (k + 1) * offset[0] + (i + 0) * offset[2];
            *(pIndex++) = (k + 1) * offset[0] + (i + 1) * offset[2];
            *(pIndex++) = (k + 0) * offset[0] + (i + 1) * offset[2];

            *(pIndex++) = (k + 0) * offset[0] + (i + 1) * offset[2] + size[0] * offset[1];
            *(pIndex++) = (k + 1) * offset[0] + (i + 1) * offset[2] + size[0] * offset[1];
            *(pIndex++) = (k + 1) * offset[0] + (i + 0) * offset[2] + size[0] * offset[1];
            *(pIndex++) = (k + 0) * offset[0] + (i + 0) * offset[2] + size[0] * offset[1];
        }

    for (int i = 0; i < size[2]; i++)
        for (int j = 0; j < size[0]; j++)
        {
            *(pIndex++) = (j + 1) * offset[1] + (i + 0) * offset[2];
            *(pIndex++) = (j + 0) * offset[1] + (i + 0) * offset[2];
            *(pIndex++) = (j + 0) * offset[1] + (i + 1) * offset[2];
            *(pIndex++) = (j + 1) * offset[1] + (i + 1) * offset[2];

            *(pIndex++) = (j + 1) * offset[1] + (i + 1) * offset[2] + size[1] * offset[0];
            *(pIndex++) = (j + 0) * offset[1] + (i + 1) * offset[2] + size[1] * offset[0];
            *(pIndex++) = (j + 0) * offset[1] + (i + 0) * offset[2] + size[1] * offset[0];
            *(pIndex++) = (j + 1) * offset[1] + (i + 0) * offset[2] + size[1] * offset[0];
        }

    return GL_TRUE;
}

unsigned char QModel::loadBinaryFile(const std::string& name)
{
    std::ifstream file(name.c_str(), std::ios_base::binary);
    if (!file)
    {
        std::cerr << " > ERROR: unable to open input file: \"" << name.c_str() << "\"." <<  std::endl;
        return GL_FALSE;
    }

    unsigned char error(0);
    error |= !QSerializer::read(file, this->name);
    error |= !QSerializerT<QModelType>::read(file, this->type);
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

unsigned char QModel::saveBinaryFile(const std::string& name)
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
    error |= !QSerializerT<QModelType>::write(file, this->type);
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

unsigned char QModel::computeVertexNormal(const unsigned char& normalized)
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
        float weight = asin(QVector3::length(cross) / (QVector3::length(edge0) * QVector3::length(edge1)));
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

unsigned char QModel::computeVertexTranslate()
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

unsigned char QModel::removeRedundantVertex()
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

unsigned char QModel::initialize()
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

unsigned char QModel::destroy()
{
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);

    return GL_TRUE;
}

unsigned char QModel::paint()
{
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

    int stride = 6 * sizeof(GLfloat);
    int normalOffset = 3 * sizeof(GLfloat);
    glVertexPointer(3, GL_FLOAT, stride, NULL);
    glNormalPointer(GL_FLOAT, stride, (GLubyte *)NULL + normalOffset);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glScalef(vertexScale, vertexScale, vertexScale);
    glTranslatef(vertexTranslate[0], vertexTranslate[1], vertexTranslate[2]);
    switch (type)
    {
    case MODEL_TRIANGLE:
        glDrawElements(GL_TRIANGLES, vertexIndexBuffer.size(), GL_UNSIGNED_INT, NULL);
    	break;
    case MODEL_QUAD:
        glDrawElements(GL_QUADS, vertexIndexBuffer.size(), GL_UNSIGNED_INT, NULL);
        break;
    }
    glPopMatrix();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    return GL_TRUE;
}

GLuint& QModel::getV()
{
    return vbo;
}

GLuint& QModel::getE()
{
    return ebo;
}

std::list<QModel>::iterator QModel::find(std::list<QModel> &models, const std::string &name)
{
    for (std::list<QModel>::iterator i = models.begin(); i != models.end(); i++)
    {
        if (i->name.find(name) != std::string::npos) return i;
    }
    return models.end();
}

void QModel::build(const std::string &path, const std::string &name, const cl_float4& modelScale)
{
    std::list<QModel> models(0);
    models.push_front(QModel(name));

    std::list<QModel>::iterator pModel = models.begin();
    pModel->type = MODEL_QUAD;
    pModel->vertexScale = 1.0f;
    pModel->vertexTranslate.assign(3, 0.0f);
    pModel->vertexNumber = 8;
    pModel->vertexBuffer.resize(pModel->vertexNumber * 6);

    float* pVertex = pModel->vertexBuffer.data();
    *(pVertex++) = -modelScale.s[0]; *(pVertex++) = -modelScale.s[1]; *(pVertex++) = -modelScale.s[2];
    *(pVertex++) = -1.0f;              *(pVertex++) = -1.0f;              *(pVertex++) = -1.0f;
    *(pVertex++) = +modelScale.s[0]; *(pVertex++) = -modelScale.s[1]; *(pVertex++) = -modelScale.s[2];
    *(pVertex++) = +1.0f;              *(pVertex++) = -1.0f;              *(pVertex++) = -1.0f;
    *(pVertex++) = -modelScale.s[0]; *(pVertex++) = +modelScale.s[1]; *(pVertex++) = -modelScale.s[2];
    *(pVertex++) = -1.0f;              *(pVertex++) = +1.0f;              *(pVertex++) = -1.0f;
    *(pVertex++) = +modelScale.s[0]; *(pVertex++) = +modelScale.s[1]; *(pVertex++) = -modelScale.s[2];
    *(pVertex++) = +1.0f;              *(pVertex++) = +1.0f;              *(pVertex++) = -1.0f;
    *(pVertex++) = -modelScale.s[0]; *(pVertex++) = -modelScale.s[1]; *(pVertex++) = +modelScale.s[2];
    *(pVertex++) = -1.0f;              *(pVertex++) = -1.0f;              *(pVertex++) = +1.0f;
    *(pVertex++) = +modelScale.s[0]; *(pVertex++) = -modelScale.s[1]; *(pVertex++) = +modelScale.s[2];
    *(pVertex++) = +1.0f;              *(pVertex++) = -1.0f;              *(pVertex++) = +1.0f;
    *(pVertex++) = -modelScale.s[0]; *(pVertex++) = +modelScale.s[1]; *(pVertex++) = +modelScale.s[2];
    *(pVertex++) = -1.0f;              *(pVertex++) = +1.0f;              *(pVertex++) = +1.0f;
    *(pVertex++) = +modelScale.s[0]; *(pVertex++) = +modelScale.s[1]; *(pVertex++) = +modelScale.s[2];
    *(pVertex++) = +1.0f;              *(pVertex++) = +1.0f;              *(pVertex++) = +1.0f;

    pModel->elementNumber = 6;
    pModel->vertexIndexBuffer.resize(pModel->elementNumber * 4);
    unsigned int* pIndex = pModel->vertexIndexBuffer.data();
    *(pIndex++) = 0; *(pIndex++) = 2; *(pIndex++) = 3; *(pIndex++) = 1;
    *(pIndex++) = 4; *(pIndex++) = 5; *(pIndex++) = 7; *(pIndex++) = 6;
    *(pIndex++) = 0; *(pIndex++) = 1; *(pIndex++) = 5; *(pIndex++) = 4;
    *(pIndex++) = 2; *(pIndex++) = 6; *(pIndex++) = 7; *(pIndex++) = 3;
    *(pIndex++) = 0; *(pIndex++) = 4; *(pIndex++) = 6; *(pIndex++) = 2;
    *(pIndex++) = 1; *(pIndex++) = 3; *(pIndex++) = 7; *(pIndex++) = 5;

    pModel->saveBinaryFile(path);
}

unsigned char QModel::getEdgeNumber(unsigned int& number)
{
    switch (type)
    {
    case MODEL_TRIANGLE:
        number = 3;
        break;
    case MODEL_QUAD:
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