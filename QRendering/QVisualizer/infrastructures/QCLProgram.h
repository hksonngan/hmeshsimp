/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QCLProgram.h
 * @brief   QCLKernel class, QCLProgram class definition.
 * 
 * QCLKernel class wraps the kernel processing functions in OpenCL.
 * QCLProgram class follows the idea of program in GLSL, which contains several kernels and memories.
 *     The kernels should perform different computing tasks.
 *     The memories which can be shared by different kernels are used for data input and output.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#ifndef QCLPROGRAM_H
#define QCLPROGRAM_H

#include <string>
#include <list>

#include "cl/cl.h"

#include "../infrastructures/QCLMemory.h"

class QCLProgram;

class QCLKernel
{
public:
    QCLKernel();
    QCLKernel(const std::string &path, const std::string &name, const std::string &entrance, const std::string &filter);
    ~QCLKernel();
    
    std::string name;
    std::string entrance;
    std::string filter;
    std::string content;
    
    unsigned char initialize(QCLProgram &program);
    unsigned char destroy();
    cl_kernel& get();
    
    static std::list<QCLKernel>::iterator find(std::list<QCLKernel> &kernels, const std::string &name);
    
private:
    cl_kernel kernel;
};

class QCLProgram
{
public:
    QCLProgram();
    QCLProgram(const std::string &name, const std::string &path);
    ~QCLProgram();
    
    std::string name;
    std::string path;
    std::list<QCLKernel> kernels;
    std::list<QCLMemory> memories;
    unsigned char initialize(const cl_context &context, const std::vector<cl_device_id> &devices, const std::string &options = std::string());
    unsigned char destroy();
    cl_program& get();

    static std::list<QCLProgram>::iterator find(std::list<QCLProgram> &programs, const std::string &name);
    
private:
    cl_program program;
};

#endif  // QCLPROGRAM_H
