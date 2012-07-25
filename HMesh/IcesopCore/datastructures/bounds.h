#ifndef ICESOP_BOUNDS_H
#define ICESOP_BOUNDS_H

#include "point.h"

#include <iostream>

namespace icesop {

/**
 * ������Χ��
 */
class Bounds
{
    unsigned int points_;   // �ӵ���Χ���еĵ�
    Point3D llf_; 	// ��(lower) ��(left) ǰ(front)
    Point3D urb_; 	// ��(upper) ��(right)��(back)

public:
    // ���캯��:δ��ȷ����
    Bounds() { points_ = 0; }

    // ���캯��:δ��ȷ����,��������v
    Bounds(const Point3D& v)
      : points_(1),
        llf_(v),
        urb_(v)
    {}

	// ���캯��:��������v1,v2
    Bounds(const Point3D& v1, const Point3D& v2)
      : points_(1),
        llf_(v1),
        urb_(v1)
    {
    	addPoint(v2);
    }

    // �����Χ��(����б�Ҫ),ʹ���������v
    void addPoint(const Point3D& v);

    // �����Χ��(����б�Ҫ),ʹ�������Χ��b
    void addVolume(const Bounds& b)
    {
        addPoint(b.llf_);
        addPoint(b.urb_);
    }

    // ��ȡLLF
    Point3D getLLF() const { return llf_; }

    // ��ȡURB
    Point3D getURB() const { return urb_; }

    // ��ȡ��Χ������
    Point3D center() const
    {
        return (diagonal() * 0.5 + llf_);
    }

    // ��ȡllf��urb�ĶԽ���
    Point3D diagonal() const
    {
        return (urb_ - llf_);
    }

    // ��ȡ��Χ�е����
    double volume() const
    {
		return abs((llf_.x - urb_.x) * (llf_.y - urb_.y) * (llf_.z - urb_.z));
    }

    // �жϰ�Χ���Ƿ񱻶���
    bool isDefined() const { return (points_ == 2); }

    // �жϰ�Χ���Ƿ�ֻ��һ����
    bool onlyPoint() const { return (points_ == 1); }

    // �жϵ�p�Ƿ��ڰ�Χ����
    bool containsPoint(const Point3D& p)
    {
        return ( (p.x >= llf_.x) && (p.y >= llf_.y) && (p.z >= llf_.z)
                    && (p.x <= urb_.x) && (p.y <= urb_.y) && (p.z <= urb_.z) );
    }

    // �ж�b�Ƿ��ڰ�Χ����
    bool containsVolume(const Bounds& b)
    {
        return ( containsPoint(b.llf_) && containsPoint(b.urb_) );
    }

    // �жϰ�Χ���Ƿ��ཻ
    // ��Χ�б����Ǳ������
	bool intersects(const Bounds& b) const
	{
        if ((llf_.x > b.urb_.x) || (b.llf_.x > urb_.x)) return false;
        if ((llf_.y > b.urb_.y) || (b.llf_.y > urb_.y)) return false;
        if ((llf_.z > b.urb_.z) || (b.llf_.z > urb_.z)) return false;

        return true;
	}

    bool insideXZ(const Bounds& b) const;
    bool insideXZ(const Point3D& v) const;
    
    // �ж�b�Ƿ��ڰ�Χ����
    bool inside(const Bounds& b) const;
    // �ж�p�Ƿ��ڰ�Χ����
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

    // ��ȡ��Χ��
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
