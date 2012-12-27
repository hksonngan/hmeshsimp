/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QProgram.cpp
 * @brief   QProgram class declaration.
 * 
 * This file declares the unified interfaces for shader processing defined in QProgram.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/16
 */

#include <gl/glew.h>

#include <iostream>

#include <QDir>

#include "../utilities/QUtility.h"
#include "../utilities/QIO.h"
#include "QProgram.h"

QAttribute::QAttribute() :
    size(0), type(0), normalized(0), stride(0), offset(0), name()
{}

QAttribute::QAttribute(const GLint &size, const GLenum &type, const GLboolean &normalized, const GLsizei &stride, const GLuint &offset, const std::string &name) :
    size(size), type(type), normalized(normalized), stride(stride), offset(offset), name(name)
{}

QAttribute::~QAttribute()
{
    this->destroy();
}

unsigned char QAttribute::activate(const GLuint &index)
{
    glVertexAttribPointer(index, size, type, normalized, stride, (char *)NULL + offset);
    if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glVertexAttribPointer()")) return GL_FALSE;

    glEnableVertexAttribArray(index);
    if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glEnableVertexAttribArray()")) return GL_FALSE;

    return GL_TRUE;
}

unsigned char QAttribute::deactivate(const GLuint &index)
{
    glDisableVertexAttribArray(index);
    if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glDisableVertexAttribArray()")) return GL_FALSE;

    return GL_TRUE;
}

unsigned char QAttribute::initialize()
{
    return GL_TRUE;
}

unsigned char QAttribute::destroy()
{
    return GL_TRUE;
}

std::list<QAttribute>::iterator QAttribute::find(std::list<QAttribute> &attributes, const std::string &name)
{
    for (std::list<QAttribute>::iterator i = attributes.begin(); i != attributes.end(); i++)
    {
        if (i->name.find(name) != std::string::npos) return i;
    }
    return attributes.end();
}

QShader::QShader() :
    type(0), name(), shader(0), content()
{}

QShader::QShader(const std::string &path, const std::string &name, GLenum type, const std::string &filter) :
    type(type), name(name), shader(0)
{
    QDir dir(path.c_str());
    QFileInfoList files = dir.entryInfoList();
    std::string content;
    for (QFileInfoList::iterator i = files.begin(); i != files.end(); i++)
    {
        std::string file = std::string((const char *)i->fileName().toLocal8Bit());
        if (file.find(filter) != std::string::npos)
        {
            if (QIO::getFileContent(path + file, content))
                this->content.append(content);
        }
    }
}

QShader::~QShader()
{
    this->destroy();
}

unsigned char QShader::initialize()
{
    shader = glCreateShader(this->type);
    if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glCreateShader()")) return GL_FALSE;

    const ::size_t size = 1;
    std::vector<const char*> sources(size, content.data());
    std::vector<GLint > lengths(size, content.size());
    glShaderSource(shader, size, sources.data(), lengths.data());
    if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glShaderSource()")) return GL_FALSE;

    return GL_TRUE;;
}

unsigned char QShader::destroy()
{
    if (this->shader)
    {
        glDeleteShader(this->shader);
        if (!QUtility::checkShaderStatus(__FILE__, __LINE__, this->shader, GL_DELETE_STATUS)) return GL_FALSE;
    }

    return GL_TRUE;
}

GLuint& QShader::get()
{
    return shader;
}

std::list<QShader>::iterator QShader::find(std::list<QShader> &shaders, const std::string &name)
{
    for (std::list<QShader>::iterator i = shaders.begin(); i != shaders.end(); i++)
    {
        if (i->name.find(name) != std::string::npos) return i;
    }
    return shaders.end();
}

QProgram::QProgram() :
    name(), path(), shaders(0), models(0), attributes(0), textures(0)
{}

QProgram::QProgram(const std::string &name, const std::string &path) :
    name(name), path(path), shaders(0), models(0), textures(0), program(0)
{}

QProgram::~QProgram()
{
    this->destroy();
}

unsigned char QProgram::activate()
{
    for (std::list<std::pair<std::list<QTexture>::iterator, GLuint>>::iterator i = this->textures.begin(); i != this->textures.end(); i++)
    {
        if (!i->first->activate(i->second)) return GL_FALSE;
    }
    for (std::list<std::pair<std::list<QAttribute>::iterator, GLuint>>::iterator i = this->attributes.begin(); i != this->attributes.end(); i++)
    {
        if (!i->first->activate(i->second)) return GL_FALSE;
    }
    glUseProgram(program);

    return GL_TRUE;
}

unsigned char QProgram::deactivate()
{
    for (std::list<std::pair<std::list<QTexture>::iterator, GLuint>>::iterator i = this->textures.begin(); i != this->textures.end(); i++)
    {
        if (!i->first->deactivate(i->second)) return GL_FALSE;
    }
    for (std::list<std::pair<std::list<QAttribute>::iterator, GLuint>>::iterator i = this->attributes.begin(); i != this->attributes.end(); i++)
    {
        if (!i->first->deactivate(i->second)) return GL_FALSE;
    }
    glUseProgram(0);

    return GL_TRUE;
}

unsigned char QProgram::setTexture(const std::list<QTexture>::iterator texture, const std::string &name, const GLenum &unit)
{
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(texture->target, texture->get());
    GLint id = glGetUniformLocation(program, name.c_str());
    if (id == -1)
    {
        std::cerr << "Error: glGetUniformLocation() - " << name.c_str() << std::endl;
        return GL_FALSE;
    }
    glUniform1i(id, unit);
    glActiveTexture(GL_TEXTURE0);

    return GL_TRUE;
}

unsigned char QProgram::setTexture(const GLenum &target, const std::string &name, const GLuint &texture, const GLenum &unit)
{
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(target, texture);
    GLint id = glGetUniformLocation(program, name.c_str());
    if (id == -1)
    {
        std::cerr << "Error: glGetUniformLocation() - " << name.c_str() << std::endl;
        return GL_FALSE;
    }
    glUniform1i(id, unit);
    glActiveTexture(GL_TEXTURE0);

    return GL_TRUE;
}

unsigned char QProgram::setUniform(const std::string &name, GLfloat* value, const int &size)
{
    GLint id = glGetUniformLocation(program, name.c_str());
    if (id == -1)
    {
        std::cerr << "Error: glGetUniformLocation() - " << name.c_str() << std::endl;
        return GL_FALSE;
    }

    switch (size)
    {
    case 1:
        glUniform1fv(id, 1, value);
        break;
    case 2:
        glUniform2fv(id, 1, value);
        break;
    case 3:
        glUniform3fv(id, 1, value);
        break;
    case 4:
        glUniform4fv(id, 1, value);
        break;
    }

    return GL_TRUE;
}

unsigned char QProgram::initialize()
{
    program = glCreateProgram();
    if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glCompileShader()")) return GL_FALSE;

    std::vector<GLchar> log(1024 * 4);
    for (std::list<QShader>::iterator i = shaders.begin(); i != shaders.end(); i++)
    {
        if (!i->initialize()) return GL_FALSE;

        glCompileShader(i->get());
        /*
        glGetShaderInfoLog(i->get(), log.size(), NULL, log.data());
        std::cout << " > LOG: [ " << i->name << " ] " << log.data() << std::endl;
        if (!QUtility::checkShaderStatus(__FILE__, __LINE__, i->get(), GL_COMPILE_STATUS)) return GL_FALSE;
        */
        glAttachShader(program, i->get());
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glAttachShader()")) return GL_FALSE;
    }

    for (std::list<std::pair<std::list<QAttribute>::iterator, GLuint>>::iterator i = attributes.begin(); i != attributes.end(); i++)
    {
        glBindAttribLocation(program, i->second, i->first->name.c_str());
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glBindAttribLocation()")) return GL_FALSE;
    }

    glLinkProgram(program);
    glGetProgramInfoLog(program, log.size(), NULL, log.data());
    std::cerr << " > LOG: [ " << name << " ] " << log.data() << std::endl;
    if (!QUtility::checkProgramStatus(__FILE__, __LINE__, program, GL_LINK_STATUS)) return GL_FALSE;

    return GL_TRUE;
}

unsigned char QProgram::destroy()
{
    for (std::list<QShader>::iterator i = shaders.begin(); i != shaders.end(); i++)
    {
        glDetachShader(program, i->get());
        if (!QUtility::checkGLStatus(__FILE__, __LINE__, "glDetachShader()")) return GL_FALSE;
    }

    if (this->program)
    {
        glDeleteProgram(this->program);
        if (!QUtility::checkProgramStatus(__FILE__, __LINE__, this->program, GL_DELETE_STATUS)) return GL_FALSE;
    }

    return GL_TRUE;
}

std::list<QProgram>::iterator QProgram::find(std::list<QProgram> &programs, const std::string &name)
{
    for (std::list<QProgram>::iterator i = programs.begin(); i != programs.end(); i++)
    {
        if (i->name.find(name) != std::string::npos) return i;
    }
    return programs.end();
}