#ifndef ICESOP_EXCEPTION_H
#define ICESOP_EXCEPTION_H

#include <string>

namespace icesop {

class IcesopException : public std::exception
{
public:
    IcesopException(const std::string& message = "") : message_(message) {}
    virtual ~IcesopException() throw() {}

    virtual const char* getMessage() const throw();
protected:
    std::string message_;
};

} // namespace icesop

#endif // ICESOP_EXCEPTION_H
