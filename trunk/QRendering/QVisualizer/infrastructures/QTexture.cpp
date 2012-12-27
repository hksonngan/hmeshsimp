/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QTexture.cpp
 * @brief   QTexture class declaration.
 * 
 * This file declares the unified interfaces for texture processing defined in QTexture.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/16
 */

#include <gl/glew.h>

#include <iostream>
#include <sstream>

#include "../utilities/QUtility.h"
#include "QTexture.h"

QTexture::QTexture() :
    target(0), level(0), internalFormat(0), border(0), format(0), type(0), name(), size(0), texture(0), buffer(NULL)
{}

QTexture::QTexture(const GLenum &target, const GLint &internalFormat, const GLenum &format, const GLenum &type, const std::string &name, const GLint &level, const GLint &border) :
    target(target), internalFormat(internalFormat), format(format), type(type), name(name), size(0), level(level), border(border), texture(0), buffer(NULL)
{}

QTexture::~QTexture()
{
    this->destroy();
}

unsigned char QTexture::activate(const GLenum &unit)
{
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(target, texture);

    return GL_TRUE;
}

unsigned char QTexture::deactivate(const GLenum &unit)
{
    glActiveTexture(GL_TEXTURE0 + unit);
    glDisable(target);

    return GL_TRUE;
}

unsigned char QTexture::read(void* buffer)
{
    switch (target)
    {
    case GL_TEXTURE_1D:
        glBindTexture(target, texture);
        glTexImage1D(target, level, internalFormat, size.at(0), border, format, type, buffer);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glTexImage1D()")) return GL_FALSE;
        break;
    case GL_TEXTURE_RECTANGLE:
        glBindTexture(target, texture);
        glTexImage2D(target, 0, internalFormat, size.at(0), size.at(1), border, GL_RGBA, GL_FLOAT, buffer);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glTexImage2D()")) return GL_FALSE;
        break;
    case GL_TEXTURE_2D:
        glBindTexture(target, texture);
        glTexImage2D(target, level, internalFormat, size.at(0), size.at(1), border, format, type, buffer);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glTexImage2D()")) return GL_FALSE;
        break;
    case GL_TEXTURE_3D:
        glBindTexture(target, texture);
        glTexImage3D(target, level, internalFormat, size.at(0), size.at(1), size.at(2), border, format, type, buffer);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glTexImage3D()")) return GL_FALSE;
        break;
    }

    return GL_TRUE;
}

unsigned char QTexture::initialize(const std::vector<GLsizei> &size, void* buffer)
{
    this->destroy();
    this->size = size;
    this->size.resize(3, 1);
    this->buffer = buffer;

    glGenTextures(1, &texture);

    switch (target)
    {
    case GL_TEXTURE_1D:
        glBindTexture(target, texture);
        glTexImage1D(target, level, internalFormat, size.at(0), border, format, type, buffer);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glTexImage1D()")) return GL_FALSE;

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        break;
    case GL_TEXTURE_RECTANGLE:
        glBindTexture(target, texture);
        glTexImage2D(target, 0, internalFormat, size.at(0), size.at(1), border, GL_RGBA, GL_FLOAT, buffer);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glTexImage2D()")) return GL_FALSE;

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        break;
    case GL_TEXTURE_2D:
        glBindTexture(target, texture);
        glTexImage2D(target, level, internalFormat, size.at(0), size.at(1), border, format, type, buffer);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glTexImage2D()")) return GL_FALSE;

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        break;
    case GL_TEXTURE_3D:
        glBindTexture(target, texture);
        glTexImage3D(target, level, internalFormat, size.at(0), size.at(1), size.at(2), border, format, type, buffer);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glTexImage3D()")) return GL_FALSE;

        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        break;
    }

    return GL_TRUE;
}

unsigned char QTexture::destroy()
{
    if (this->texture)
    {
        glDeleteTextures(1, &this->texture);
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glDeleteTextures()")) return GL_FALSE;
    }

    return GL_TRUE;
}

GLuint& QTexture::get()
{
    return texture;
}

std::list<QTexture>::iterator QTexture::find(std::list<QTexture> &textures, const std::string &name)
{
    for (std::list<QTexture>::iterator i = textures.begin(); i != textures.end(); i++)
    {
        if (i->name.find(name) != std::string::npos) return i;
    }
    return textures.end();
}

/*
void QTexture::updateTexture(std::vector<QTexture> &textures, int index, void *data)
{
    QTexture *texture = &textures.at(index);
    unsigned int id = texture->id;
    unsigned int target = -1;
    int dimension = texture->size.size();
    switch (dimension)
    {
    case 1:
        target = GL_TEXTURE_1D;
        glBindTexture(target, id);
        glTexSubImage1D(target, 0, 0, texture->size.at(0), texture->format, texture->type, data);
        break;
    case 2:
        target = GL_TEXTURE_2D;
        glBindTexture(target, id);
        glTexSubImage2D(target, 0, 0, 0, texture->size.at(0), texture->size.at(1), texture->format, texture->type, data);
        break;
    case 3:
        target = GL_TEXTURE_3D;
        glBindTexture(GL_TEXTURE_3D, id);
        glTexSubImage3D(target, 0, 0, 0, 0, texture->size.at(0), texture->size.at(1), texture->size.at(2), texture->format, texture->type, data);
        break;
    }
}
*/