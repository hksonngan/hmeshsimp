#include "exception.h"

namespace icesop {

const char* IcesopException::getMessage() const throw()
{
    return message_.c_str();
}

} // namespace icesop