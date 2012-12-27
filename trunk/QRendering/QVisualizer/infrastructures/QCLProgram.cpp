/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QCLProgram.cpp
 * @brief   QCLKernel class, QCLProgram class declaration.
 * 
 * This file declares the commonly used methods defined in QCLProgram.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include <gl/glew.h>

#include <iostream>

#include <QDIR>

#include "../utilities/QIO.h"
#include "../utilities/QUtility.h"
#include "QCLProgram.h"

QCLKernel::QCLKernel() :
    name(), entrance(), filter(), content(), kernel(0)
{}

QCLKernel::QCLKernel(const std::string &path, const std::string &name, const std::string &entrance, const std::string &filter) :
    name(name), entrance(entrance), filter(filter), kernel(0)
{
    QDir dir(path.c_str());
    QFileInfoList list = dir.entryInfoList();
    for (QFileInfoList::iterator i = list.begin(); i != list.end(); i++)
    {
        std::string name = std::string((const char *)i->fileName().toLocal8Bit());
        if (name.find(filter) != std::string::npos)
        {
            std::string source;
            QIO::getFileContent(path + name, source);
            this->content += source;
        }
    }
}

QCLKernel::~QCLKernel()
{
    this->destroy();
}

unsigned char QCLKernel::initialize(QCLProgram &program)
{
    this->destroy();

    cl_int status = CL_SUCCESS;
    kernel = clCreateKernel(program.get(), entrance.c_str(), &status);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateKernel()")) return GL_FALSE;

    return GL_TRUE;
}

unsigned char QCLKernel::destroy()
{
    if (this->kernel)
    {
        cl_int status = clReleaseKernel(this->kernel);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clReleaseKernel()")) return GL_FALSE;
        this->kernel = 0;
    }

    return GL_TRUE;
}

cl_kernel& QCLKernel::get()
{
    return kernel;
}

std::list<QCLKernel>::iterator QCLKernel::find(std::list<QCLKernel> &kernels, const std::string &name)
{
    for (std::list<QCLKernel>::iterator i = kernels.begin(); i != kernels.end(); i++)
    {
        if (i->name.find(name) != std::string::npos) return i;
    }
    return kernels.end();
}

QCLProgram::QCLProgram() :
    name(), path(), kernels(), memories(), program(0)
{}

QCLProgram::QCLProgram(const std::string &name, const std::string &path) :
    name(name), path(path), kernels(), memories(), program(0)
{}

QCLProgram::~QCLProgram()
{
    this->destroy();
}

unsigned char QCLProgram::initialize(const cl_context &context, const std::vector<cl_device_id> &devices, const std::string &options)
{
    this->destroy();

    cl_int status = CL_SUCCESS;

    std::string source;
    for (std::list<QCLKernel>::const_iterator i = kernels.begin(); i != kernels.end(); i++)
        source += i->content;

    const ::size_t size = 1;
    std::vector<const char*> sources(size, source.data());
    std::vector<::size_t> lengths(size, source.size());
    program = clCreateProgramWithSource(context, size, sources.data(), lengths.data(), &status);
    if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clCreateProgramWithSource()")) return GL_FALSE;

    status = clBuildProgram(program, devices.size(), (cl_device_id*)devices.data(), options.c_str(), NULL, NULL);
    std::vector<unsigned char> log(1024 * 4);
    for (std::vector<cl_device_id>::const_iterator i = devices.begin(); i != devices.end(); i++)
    {
        status = clGetProgramBuildInfo(program, *i, CL_PROGRAM_BUILD_LOG, log.size() - 1, log.data(), NULL);
        std::cout << " > LOG: [ device " << i - devices.begin() << " ] " << log.data() << std::endl;
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clGetProgramBuildInfo()")) return GL_FALSE;
    }

    for (std::list<QCLKernel>::iterator i = kernels.begin(); i != kernels.end(); i++)
    {
        if (!i->initialize(*this)) return GL_FALSE;
    }

    return GL_TRUE;
}

cl_program& QCLProgram::get()
{
    return program;
}

unsigned char QCLProgram::destroy()
{
    if (this->program)
    {
        cl_int status = clReleaseProgram(this->program);
        if (!QUtility::checkCLStatus(__FILE__, __LINE__, status, "clReleaseProgram()")) return GL_FALSE;
        this->program = 0;
    }

    return GL_TRUE;
}

std::list<QCLProgram>::iterator QCLProgram::find(std::list<QCLProgram> &programs, const std::string &name)
{
    for (std::list<QCLProgram>::iterator i = programs.begin(); i != programs.end(); i++)
    {
        if (i->name.find(name) != std::string::npos) return i;
    }
    return programs.end();
}