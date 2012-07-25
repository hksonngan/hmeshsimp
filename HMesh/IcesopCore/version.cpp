
#include "version.h"

#include <sstream>

namespace icesop {

const std::string IcesopVersion::getCompilerVersion()
{
    std::stringstream ver;
#if defined(WIN32) && defined(_MSC_VER)
    #if _MSC_VER == 1310
        ver << "Microsoft Visual C++ .NET 2003";
    #elif _MSC_VER == 1400
        ver << "Microsoft Visual C++ 2005";
    #elif _MSC_VER == 1500
        ver << "Microsoft Visual C++ 2008";
    #elif _MSC_VER == 1600
        ver << "Microsoft Visual C++ 2010";
    #else
        ver << "Unknown Microsoft Visual C++ (_MSC_VER=" << _MSC_VER << ")";
    #endif
#elif defined(__GNUC__)
    ver << "gcc " << __VERSION__;
#else
    ver << "Unknown compiler";
#endif

#if defined(__amd64__) || defined(__x86_64__) || defined(_M_X64)
    ver << " [x86_64]";
#elif defined(__i386__) || defined(_M_IX86) || defined(_X86_) || defined(__INTEL__)
    ver << " [x86]";
#elif defined(__powerpc__)
    ver << " [PowerPC]";
#else
    ver << " [unknown CPU architecture]";
#endif

    return ver.str();
}

const std::string IcesopVersion::getVersion()
{
    return "0.1.0";
}


const std::string IcesopVersion::getCopyright()
{
    return "Copyright (C) 2011-2014 THSS.CGCAD.";
}

} // namespace icesop
