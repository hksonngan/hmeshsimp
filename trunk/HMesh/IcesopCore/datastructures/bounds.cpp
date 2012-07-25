#include "bounds.h"

#include <algorithm>

namespace icesop {

using std::min;
using std::max;

void Bounds::addPoint(const Point3D& v)
{
    if (points_ < 2)
   	{
        ++points_;
        if (points_ == 1)
        {
            llf_ = v; urb_ = v;
            return;
        }
    }

    llf_ = Min(llf_, v);
    urb_ = Max(urb_, v);
}

bool Bounds::insideXZ(const Bounds& b) const
{
    //Assert(  isDefined(), "This Box is not defined.");
    //Assert(b.isDefined(), "Box b is not defined.");

    Point3D llfb = b.getLLF();
    Point3D urbb = b.getURB();

    double r0x = min(llf_[0], urb_[0]);
    double r1x = max(llf_[0], urb_[0]);
    double r0y = min(llf_[2], urb_[2]);
    double r1y = max(llf_[2], urb_[2]);
    double r2x = min(llfb[0], urbb[0]);
    double r3x = max(llfb[0], urbb[0]);
    double r2y = min(llfb[2], urbb[2]);
    double r3y = max(llfb[2], urbb[2]);

    return (r0x >= r2x) && (r0y >= r2y)
        && (r1x <= r3x) && (r1y <= r3y);
}

bool Bounds::insideXZ(const Point3D& v) const
{
    //Assert(isDefined(), "This Box is not defined.");

    return (llf_[0] <= v[0]) && (v[0] <= urb_[0])
        && (llf_[2] <= v[2]) && (v[2] <= urb_[2]);
}

bool Bounds::inside(const Bounds& b) const
{
    //Assert(  isDefined(), "This Box is not defined.");
    //Assert(b.isDefined(), "Box b is not defined.");

    Point3D llfb = b.getLLF();
    Point3D urbb = b.getURB();

    double r0x = min(llf_[0], urb_[0]);
    double r1x = max(llf_[0], urb_[0]);
    double r0y = min(llf_[1], urb_[1]);
    double r1y = max(llf_[1], urb_[1]);
    double r0z = min(llf_[2], urb_[2]);
    double r1z = max(llf_[2], urb_[2]);
    
    double r2x = min(llfb[0], urbb[0]);
    double r3x = max(llfb[0], urbb[0]);
    double r2y = min(llfb[1], urbb[1]);
    double r3y = max(llfb[1], urbb[1]);
    double r2z = min(llfb[2], urbb[2]);
    double r3z = max(llfb[2], urbb[2]);

    return (r0x >= r2x) && (r1x <= r3x)
        && (r0y >= r2y) && (r1y <= r3y)
        && (r0z >= r2z) && (r1z <= r3z);
}

bool Bounds::inside(const Point3D& v) const
{
    //Assert(isDefined(), "This Box ist not defined.");

    return (llf_[0] <= v[0]) && (v[0] <= urb_[0])
        && (llf_[1] <= v[1]) && (v[1] <= urb_[1])
        && (llf_[2] <= v[2]) && (v[2] <= urb_[2]);
}

std::ostream& operator<< (std::ostream& o, const Bounds& b)
{
    return (o << "(llf: " << b.getLLF().x << "," << b.getLLF().y << "," << b.getLLF().z 
			  << " urb: " << b.getURB().x << "," << b.getURB().y << "," << b.getURB().z << ")");
}

} // namespace icesop

