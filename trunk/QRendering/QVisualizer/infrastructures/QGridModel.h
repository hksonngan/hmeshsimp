/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QGridModel.h
 * @brief   QGridModel class definition.
 * 
 * This file defines ...
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/04/14
 */

#ifndef QGRIDMODEL_H
#define QGRIDMODEL_H

#include <string>
#include <vector>
#include <list>

enum QGridModelType
{
    MODEL_UNKNOWN       = 0,
    MODEL_TETRAHEDRON   = 1,
    MODEL_HEXAHEDRON    = 2,
    MODEL_HYBRID        = 3
};

class QGridModel
{
public:
    QGridModel();
    QGridModel(const std::string &name);
    ~QGridModel();
    
    QGridModelType type;
    std::string name;
    float vertexScale;
    unsigned int vertexNumber, elementNumber;
    std::vector<float> vertexTranslate;
    std::vector<float> vertexBuffer;
    std::vector<unsigned int> vertexIndexBuffer;
    
    unsigned char loadBinaryFile(const std::string& name);
    unsigned char saveBinaryFile(const std::string& name);
    unsigned char computeVertexTranslate();
    unsigned char computeVertexNormal(const unsigned char& normalized = 0);
    unsigned char removeRedundantVertex();

    unsigned char initialize();
    unsigned char destroy();
    GLuint& getV();
    GLuint& getE();

    static std::list<QGridModel>::iterator find(std::list<QGridModel> &models, const std::string &name);

private:
    GLuint vbo;
    GLuint ebo;
    
    unsigned char getEdgeNumber(unsigned int& number);
};

#endif  // QGRIDMODEL_H