#ifndef ICESOP_BOUNDS_H
#define ICESOP_BOUNDS_H

#include "point.h"

#include <iostream>

namespace icesop {

/**
 * 轴对齐包围盒
 */
class Bounds
{
    unsigned int points_;   // 加到包围盒中的点
    Point3D llf_; 	// 下(lower) 左(left) 前(front)
    Point3D urb_; 	// 上(upper) 右(right)后(back)

public:
    // 构造函数:未明确定义
    Bounds() { points_ = 0; }

    // 构造函数:未明确定义,包含向量v
    Bounds(const Point3D& v)
      : points_(1),
        llf_(v),
        urb_(v)
    {}

	// 构造函数:包含向量v1,v2
    Bounds(const Point3D& v1, const Point3D& v2)
      : points_(1),
        llf_(v1),
        urb_(v1)
    {
    	addPoint(v2);
    }

    // 扩大包围盒(如果有必要),使其包含向量v
    void addPoint(const Point3D& v);

    // 扩大包围盒(如果有必要),使其包含包围盒b
    void addVolume(const Bounds& b)
    {
        addPoint(b.llf_);
        addPoint(b.urb_);
    }

    // 获取LLF
    Point3D getLLF() const { return llf_; }

    // 获取URB
    Point3D getURB() const { return urb_; }

    // 获取包围盒中心
    Point3D center() const
    {
        return (diagonal() * 0.5 + llf_);
    }

    // 获取llf到urb的对角线
    Point3D diagonal() const
    {
        return (urb_ - llf_);
    }

    // 获取包围盒的体积
    double volume() const
    {
		return abs((llf_.x - urb_.x) * (llf_.y - urb_.y) * (llf_.z - urb_.z));
    }

    // 判断包围盒是否被定义
    bool isDefined() const { return (points_ == 2); }

    // 判断包围盒是否只有一个点
    bool onlyPoint() const { return (points_ == 1); }

    // 判断点p是否在包围盒内
    bool containsPoint(const Point3D& p)
    {
        return ( (p.x >= llf_.x) && (p.y >= llf_.y) && (p.z >= llf_.z)
                    && (p.x <= urb_.x) && (p.y <= urb_.y) && (p.z <= urb_.z) );
    }

    // 判断b是否在包围盒内
    bool containsVolume(const Bounds& b)
    {
        return ( containsPoint(b.llf_) && containsPoint(b.urb_) );
    }

    // 判断包围盒是否相交
    // 包围盒必须是被定义的
	bool intersects(const Bounds& b) const
	{
        if ((llf_.x > b.urb_.x) || (b.llf_.x > urb_.x)) return false;
        if ((llf_.y > b.urb_.y) || (b.llf_.y > urb_.y)) return false;
        if ((llf_.z > b.urb_.z) || (b.llf_.z > urb_.z)) return false;

        return true;
	}

    bool insideXZ(const Bounds& b) const;
    bool insideXZ(const Point3D& v) const;
    
    // 判断b是否在包围盒内
    bool inside(const Bounds& b) const;
    // 判断p是否在包围盒内
    bool inside(const Point3D& v) const;
};

class HasBounds
{
public:
    HasBounds(const Bounds& bounds)
      : boundingBox_(bounds)
    {}

    HasBounds()
      : boundingBox_(Bounds())
    {}

    // 获取包围盒
    const Bounds& getBounds() const
    {
        return boundingBox_;
    }

protected:
    Bounds boundingBox_;
};

std::ostream& operator<< (std::ostream& o, const Bounds& b);

} // namespace icesop

#endif // ICESOP_BOUNDS_H
