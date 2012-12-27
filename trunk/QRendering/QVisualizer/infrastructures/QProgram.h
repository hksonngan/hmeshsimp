/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QProgram.h
 * @brief   QAttribute class, QShader class, QProgram class definition.
 * 
 * QAttribute class ...
 * QShader class ...
 * QProgram class ...
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/16
 */

#ifndef QPROGRAM_H
#define QPROGRAM_H

#include <string>
#include <vector>
#include <list>

#include "QStructure.h"
#include "QTexture.h"

class QModel;

class QAttribute
{
public:
    QAttribute();
    QAttribute(const GLint &size, const GLenum &type, const GLboolean &normalized, const GLsizei &stride, const GLuint &offset, const std::string &name);
    ~QAttribute();

    GLint size;
    GLenum type;
    GLboolean normalized;
    GLsizei stride;
    GLuint offset;
    std::string name;

    unsigned char activate(const GLuint &index);
    unsigned char deactivate(const GLuint &index);

    unsigned char initialize();
    unsigned char destroy();
    GLuint& get();

    static std::list<QAttribute>::iterator find(std::list<QAttribute> &attributes, const std::string &name);
};

class QShader
{
public:
    QShader();
    QShader(const std::string &path, const std::string &name, GLenum type, const std::string &filter);
    ~QShader();

    GLenum type;
    std::string name;
    std::string content;
    
    unsigned char initialize();
    unsigned char destroy();
    GLuint& get();

    static std::list<QShader>::iterator find(std::list<QShader> &shaders, const std::string &name);

private:
    GLuint shader;
};

class QProgram
{
public:
    QProgram();
    QProgram(const std::string &name, const std::string &path);
    ~QProgram();
    
    std::string name, path;
    std::list<QShader> shaders;
    std::list<std::list<QModel>::iterator> models;
    std::list<std::pair<std::list<QAttribute>::iterator, GLuint>> attributes;
    std::list<std::pair<std::list<QTexture>::iterator, GLuint>> textures;
    
    unsigned char activate();
    unsigned char deactivate();

    unsigned char setTexture(const GLenum &target, const std::string &name, const GLuint &texture, const GLenum &unit);
    unsigned char setTexture(const std::list<QTexture>::iterator texture, const std::string &name, const GLenum &unit);
    unsigned char setUniform(const std::string &name, GLfloat* value, const int &size);
    
    unsigned char initialize();
    unsigned char destroy();
    
    static std::list<QProgram>::iterator find(std::list<QProgram> &programs, const std::string &name);

private:
    GLuint program;
};

#endif  // QPROGRAM
