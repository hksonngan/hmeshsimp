/****************************************************************************
** Meta object code from reading C++ file 'QVRWidget.h'
**
** Created: Tue Apr 17 11:29:32 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../VolumeRendering/QVRWidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QVRWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QVRWidget[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
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
     225,   10,  145,   10, 0x0a,
     246,   10,  145,   10, 0x0a,
     271,   10,  145,   10, 0x0a,
     295,   10,  145,   10, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_QVRWidget[] = {
    "QVRWidget\0\0histogramData,histogramSize\0"
    "signalHistogramInitialized(float*,::size_t)\0"
    "histogramID,histogramData\0"
    "signalHistogramUpdated(uint,float*)\0"
    "unsigned char\0controlPoints,width\0"
    "slotUpdateTransferFunction(QHoverPoints*,int)\0"
    "slotUpdateStepSize()\0slotUpdateVolumeOffset()\0"
    "slotUpdateVolumeScale()\0slotUpdateTimeStep()\0"
};

const QMetaObject QVRWidget::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_QVRWidget,
      qt_meta_data_QVRWidget, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QVRWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QVRWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QVRWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QVRWidget))
        return static_cast<void*>(const_cast< QVRWidget*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int QVRWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: signalHistogramInitialized((*reinterpret_cast< float*(*)>(_a[1])),(*reinterpret_cast< ::size_t(*)>(_a[2]))); break;
        case 1: signalHistogramUpdated((*reinterpret_cast< uint(*)>(_a[1])),(*reinterpret_cast< float*(*)>(_a[2]))); break;
        case 2: { unsigned char _r = slotUpdateTransferFunction((*reinterpret_cast< QHoverPoints*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        case 3: { unsigned char _r = slotUpdateStepSize();
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        case 4: { unsigned char _r = slotUpdateVolumeOffset();
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        case 5: { unsigned char _r = slotUpdateVolumeScale();
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        case 6: { unsigned char _r = slotUpdateTimeStep();
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        default: ;
        }
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void QVRWidget::signalHistogramInitialized(float * _t1, ::size_t _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void QVRWidget::signalHistogramUpdated(unsigned int _t1, float * _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
