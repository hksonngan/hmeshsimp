/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QIO.cpp
 * @brief   QIO class declaration.
 * 
 * This file declares the most commonly used methods defined in QIO.h.
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/02/07
 */

#include <gl/glew.h>

#include <iostream>
#include <fstream>

#include "QUtility.h"
#include "QIO.h"

QIO::QIO()
{}

QIO::~QIO()
{}

unsigned char QIO::getFileContent(std::string fileName, std::string &content)
{
    std::ifstream file(fileName.c_str());
    if (!file)
    {
        std::cerr << " > ERROR: unable to open input file: \"" << fileName << "\"." <<  std::endl;
        return GL_FALSE;
    }
    
    file.seekg(0, std::ios::end);
    int length = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // content.reserve(length);
    // content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    content.resize(length);
    file.read((char*)content.data(), length);
    file.close();
    
    return GL_TRUE;
}

unsigned char QIO::getFileData(std::string fileName, void *data, unsigned int size)
{
    std::ifstream file(fileName.c_str(), std::ios::binary);
    if (!file)
    {
        std::cerr << " > ERROR: unable to open input file: \"" << fileName << "\"." <<  std::endl;
        return GL_FALSE;
    }
    
    file.read((char *)data, size);
    if (file.gcount() != size)
    {
        std::cerr << " > ERROR: reading data failed." << std::endl;
        return GL_FALSE;
    }
    file.close();

    return GL_TRUE;
}

// read file from the offset, -ht
unsigned char QIO::getFileDataOff(std::string fileName, void *data, unsigned int size, int offset)
{
	std::ifstream file(fileName.c_str(), std::ios::binary);
	if (!file)
	{
		std::cerr << " > ERROR: unable to open input file: \"" << fileName << "\"." <<  std::endl;
		return GL_FALSE;
	}

	file.seekg(offset);

	file.read((char *)data, size);
	if (file.gcount() != size)
	{
		std::cerr << " > ERROR: reading data failed." << std::endl;
		return GL_FALSE;
	}
	file.close();

	return GL_TRUE;
}

unsigned char saveFileData(std::string fileName, void *data, unsigned int size)
{
    std::ofstream file(fileName.c_str(), std::ios::binary);
    if (!file)
    {
        std::cerr << " > ERROR: unable to open output file: \"" << fileName << "\"." <<  std::endl;
        return GL_FALSE;
    }

    file.write((char *)data, size);
    if (file.fail())
    {
        std::cerr << " > ERROR: writing data failed." << std::endl;
        return GL_FALSE;
    }
    file.close();

    return GL_TRUE; 
}