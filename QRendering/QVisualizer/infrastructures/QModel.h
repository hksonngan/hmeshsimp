/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QModel.h
 * @brief   QModel class definition.
 * 
 * This file defines ...
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/19
 */

#ifndef QMODEL_H
#define QMODEL_H

#include <string>
#include <vector>
#include <list>

enum QModelType
{
    MODEL_UNKNOWN    = 0,
    MODEL_TRIANGLE   = 1,
    MODEL_QUAD       = 2,
    MODEL_HYBRID     = 3
};

class QModel
{
public:
    QModel();
    QModel(const std::string &name);
    ~QModel();
    
    QModelType type;
    std::string name;
    float vertexScale;
    unsigned int vertexNumber, elementNumber;
    std::vector<float> vertexTranslate;
    std::vector<float> vertexBuffer;
    std::vector<unsigned int> vertexIndexBuffer;
    
    unsigned char loadGemFile(const std::string& name);
    unsigned char loadBinaryFile(const std::string& name);
    unsigned char saveBinaryFile(const std::string& name);
    unsigned char computeVertexTranslate();
    unsigned char computeVertexNormal(const unsigned char& normalized = 0);
    unsigned char removeRedundantVertex();

    unsigned char initialize();
    unsigned char destroy();
    unsigned char paint();
    GLuint& getV();
    GLuint& getE();

    static std::list<QModel>::iterator find(std::list<QModel> &models, const std::string &name);
    static void build(const std::string &path, const std::string &name, const cl_float4& modelScale);

private:
    GLuint vbo;
    GLuint ebo;
    
    unsigned char getEdgeNumber(unsigned int& number);
};

#endif  // QMODEL_H