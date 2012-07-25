
#ifndef ICESOP_VERSION_H
#define ICESOP_VERSION_H

#include <string>

namespace icesop {

class IcesopVersion {
public:
    static const std::string getCompilerVersion();
    static const std::string getVersion();
    static const std::string getCopyright();
};

} // namespace icesop

#endif // ICESOP_VERSION_H
