/****************************************************************************
** Meta object code from reading C++ file 'QDPWidget.h'
**
** Created: Thu Dec 13 14:28:39 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../DepthPeeling/QDPWidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QDPWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QDPWidget[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: signature, parameters, type, tag, flags
      39,   11,   10,   10, 0x05,
     109,   83,   10,   10, 0x05,

 // slots: signature, parameters, type, tag, flags
     179,  159,  145,   10, 0x0a,
     231,  225,  145,   10, 0x0a,
     255,  225,  145,   10, 0x0a,
     283,  225,  145,   10, 0x0a,
     310,  225,  145,   10, 0x0a,
     334,  225,  145,   10, 0x0a,
     355,  225,  145,   10, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_QDPWidget[] = {
    "QDPWidget\0\0histogramSize,histogramData\0"
    "signalHistogramInitialized(::size_t,float*)\0"
    "histogramID,histogramData\0"
    "signalHistogramUpdated(uint,float*)\0"
    "unsigned char\0controlPoints,width\0"
    "slotUpdateTransferFunction(QHoverPoints*,int)\0"
    "value\0slotUpdateStepSize(int)\0"
    "slotUpdateVolumeOffset(int)\0"
    "slotUpdateVolumeScale(int)\0"
    "slotUpdateTimeStep(int)\0slotUpdateColor(int)\0"
    "slotUpdateAlpha(int)\0"
};

const QMetaObject QDPWidget::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_QDPWidget,
      qt_meta_data_QDPWidget, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QDPWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QDPWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QDPWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QDPWidget))
        return static_cast<void*>(const_cast< QDPWidget*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int QDPWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: signalHistogramInitialized((*reinterpret_cast< ::size_t(*)>(_a[1])),(*reinterpret_cast< float*(*)>(_a[2]))); break;
        case 1: signalHistogramUpdated((*reinterpret_cast< uint(*)>(_a[1])),(*reinterpret_cast< float*(*)>(_a[2]))); break;
        case 2: { unsigned char _r = slotUpdateTransferFunction((*reinterpret_cast< QHoverPoints*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        case 3: { unsigned char _r = slotUpdateStepSize((*reinterpret_cast< int(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        case 4: { unsigned char _r = slotUpdateVolumeOffset((*reinterpret_cast< int(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        case 5: { unsigned char _r = slotUpdateVolumeScale((*reinterpret_cast< int(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        case 6: { unsigned char _r = slotUpdateTimeStep((*reinterpret_cast< int(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        case 7: { unsigned char _r = slotUpdateColor((*reinterpret_cast< int(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        case 8: { unsigned char _r = slotUpdateAlpha((*reinterpret_cast< int(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        default: ;
        }
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void QDPWidget::signalHistogramInitialized(::size_t _t1, float * _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void QDPWidget::signalHistogramUpdated(unsigned int _t1, float * _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
