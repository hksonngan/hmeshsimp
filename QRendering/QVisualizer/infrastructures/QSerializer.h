/**
 * COPYRIGHT NOTICE
 * Copyright (c) 2012, Institute of CG & CAD, Tsinghua University.
 * All Rights Reserved.
 * 
 * @file    QSerializer.h
 * @brief   QSerializer class definition.
 * 
 * This file defines ...
 * 
 * @version 1.0
 * @author  Jackie Pang
 * @e-mail: 15pengyi@gmail.com
 * @date    2012/03/20
 */

#ifndef QSERIALIZER_H
#define QSERIALIZER_H

#include <gl/glew.h>

#include <iostream>
#include <vector>
#include <string>

class QSerializer
{
public:
    static unsigned char read(std::istream &s, std::string &v)
    {
        int size = 0;
        s.read((char *)&size, sizeof(int));
        if (s.gcount() != sizeof(int)) return GL_FALSE;

        if (size)
        {
            v.resize(size);
            if (v.size() != size) return GL_FALSE;

            s.read((char *)v.data(), size);
            if (s.gcount() != size) return GL_FALSE;
        }

        return GL_TRUE;
    }

    static unsigned char write(std::ostream &s, const std::string &v)
    {
        int size = v.size();
        s.write((char *)&size, sizeof(int));
        if (size)
        {
            s.write((char *)v.data(), size);
        }

        return GL_TRUE;
    }
};

template<typename T>
class QSerializerT
{
public:
    static unsigned char read(std::istream &s, T &v)
    {
        int size = sizeof(T);
        s.read((char *)&v, size);
        if (s.gcount() != size) return GL_FALSE;

        return GL_TRUE;
    }

    static unsigned char read(std::istream &s, std::vector<T> &v, unsigned char hybrid = 0)
    {
        int size = 0;
        s.read((char *)&size, sizeof(int));
        if (s.gcount() != sizeof(int)) return GL_FALSE;

        if (size)
        {
            v.resize(size);
            if (v.size() != size) return GL_FALSE;

            if (!hybrid)
            {
                size *= sizeof(T);
                s.read((char *)v.data(), size);
                if (s.gcount() != size) return GL_FALSE;
            }
            else
            {
                for (std::vector<T>::iterator i = v.begin(); i != v.end(); i++)
                    if (!QSerializerT<T>::read(s, *i)) return GL_FALSE;
            }
        }

        return GL_TRUE;
    };

    static unsigned char write(std::ostream &s, const T &v)
    {
        int size = sizeof(T);
        s.write((char *)&v, size);

        return GL_TRUE;
    };

    static unsigned char write(std::ostream &s, const std::vector<T> &v, unsigned char hybrid = 0)
    {
        int size = v.size();
        s.write((char *)&size, sizeof(int));

        if (size)
        {
            if (!hybrid)
            {
                s.write((char *)v.data(), size * sizeof(T));
            }
            else
            {
                for (std::vector<T>::const_iterator i = v.begin(); i != v.end(); i++)
                    QSerializerT<T>::write(s, *i);
            }
        }

        return GL_TRUE;
    };
};

#endif  // QSERIALIZER_H