/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QTexture.h
 * @brief   QTexture class definition.
 * 
 * QTexture class ...
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/16
 */

#ifndef QTEXTURE_H
#define QTEXTURE_H

#include <vector>
#include <list>
#include <string>

class QTexture
{
public:
    QTexture();
    QTexture(const GLenum &target, const GLint &internalFormat, const GLenum &format, const GLenum &type, const std::string &name, const GLint &level = 0, const GLint &border = 0);
    ~QTexture();

    GLenum target;
    GLint level;
    GLint internalFormat;
    GLint border;
    GLenum format;
    GLenum type;
    std::string name;
    std::vector<GLsizei> size;
    
    unsigned char activate(const GLenum &unit);
    unsigned char deactivate(const GLenum &unit);

    unsigned char read(void* buffer);
    unsigned char initialize(const std::vector<GLsizei> &size, void* buffer = NULL);
    unsigned char destroy();
    GLuint& get();
    const void* getBuffer();

    // static void updateTexture(std::vector<QTexture> &textures, int index, void *data);

    static std::list<QTexture>::iterator find(std::list<QTexture> &textures, const std::string &name);

private:
    GLuint texture;
    void* buffer;

};

#endif  // QTEXTURE_H