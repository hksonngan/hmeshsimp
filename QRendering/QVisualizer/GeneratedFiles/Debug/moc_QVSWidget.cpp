/****************************************************************************
** Meta object code from reading C++ file 'QVSWidget.h'
**
** Created: Wed Jan 2 15:03:22 2013
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../ViewSelection/QVSWidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QVSWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QVSWidget[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
      30,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       9,       // signalCount

 // signals: signature, parameters, type, tag, flags
      39,   11,   10,   10, 0x05,
     109,   83,   10,   10, 0x05,
     188,  145,   10,   10, 0x05,
     254,  145,   10,   10, 0x05,
     332,  320,   10,   10, 0x05,
     371,  320,   10,   10, 0x05,
     418,  410,   10,   10, 0x05,
     451,   10,   10,   10, 0x05,
     475,   10,   10,   10, 0x05,

 // slots: signature, parameters, type, tag, flags
     528,  499,   10,   10, 0x0a,
     594,  588,   10,   10, 0x0a,
     624,  588,   10,   10, 0x0a,
     652,  588,   10,   10, 0x0a,
     679,  588,   10,   10, 0x0a,
     717,  588,   10,   10, 0x0a,
     746,  588,   10,   10, 0x0a,
     778,  588,   10,   10, 0x0a,
     810,  588,   10,   10, 0x0a,
     843,  588,   10,   10, 0x0a,
     876,  588,   10,   10, 0x0a,
     915,  909,   10,   10, 0x0a,
     944,  588,   10,   10, 0x0a,
     980,  588,   10,   10, 0x0a,
    1016,  588,   10,   10, 0x0a,
    1053,  588,   10,   10, 0x0a,
    1086,   10,   10,   10, 0x0a,
    1111,   10,   10,   10, 0x0a,
    1136,   10,   10,   10, 0x0a,
    1161,   10,   10,   10, 0x0a,
    1182,   10,   10,   10, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_QVSWidget[] = {
    "QVSWidget\0\0histogramSize,histogramData\0"
    "signalHistogramInitialized(::size_t,float*)\0"
    "histogramID,histogramData\0"
    "signalHistogramUpdated(uint,float*)\0"
    "viewSize,viewEntropy,minEntropy,maxEntropy\0"
    "signalNorthernViewEntropyInitialized(::size_t,float*,float,float)\0"
    "signalSouthernViewEntropyInitialized(::size_t,float*,float,float)\0"
    "type,offset\0signalNorthernViewPointMarked(int,int)\0"
    "signalSouthernViewPointMarked(int,int)\0"
    "entropy\0signalViewEntropyUpdated(double)\0"
    "signalLoadViewEntropy()\0signalSaveViewEntropy()\0"
    "controlPoints,width,modified\0"
    "slotUpdateTransferFunction(QHoverPoints*,int,unsigned char)\0"
    "value\0slotUpdateVolumeStepSize(int)\0"
    "slotUpdateVolumeOffset(int)\0"
    "slotUpdateVolumeScale(int)\0"
    "slotUpdateComputingEntropyState(bool)\0"
    "slotUpdateShadingState(bool)\0"
    "slotUpdateGaussian1DState(bool)\0"
    "slotUpdateGaussian2DState(bool)\0"
    "slotUpdateLightPositionX(double)\0"
    "slotUpdateLightPositionY(double)\0"
    "slotUpdateLightPositionZ(double)\0color\0"
    "slotUpdateLightColor(QColor)\0"
    "slotUpdateLightDiffuseCoeff(double)\0"
    "slotUpdateLightAmbientCoeff(double)\0"
    "slotUpdateLightSpecularCoeff(double)\0"
    "slotUpdateMaterialShininess(int)\0"
    "slotLoadConfigurations()\0"
    "slotSaveConfigurations()\0"
    "slotComputeViewEntropy()\0slotMarkStartPoint()\0"
    "slotMarkEndPoint()\0"
};

const QMetaObject QVSWidget::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_QVSWidget,
      qt_meta_data_QVSWidget, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QVSWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QVSWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QVSWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QVSWidget))
        return static_cast<void*>(const_cast< QVSWidget*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int QVSWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: signalHistogramInitialized((*reinterpret_cast< ::size_t(*)>(_a[1])),(*reinterpret_cast< float*(*)>(_a[2]))); break;
        case 1: signalHistogramUpdated((*reinterpret_cast< uint(*)>(_a[1])),(*reinterpret_cast< float*(*)>(_a[2]))); break;
        case 2: signalNorthernViewEntropyInitialized((*reinterpret_cast< ::size_t(*)>(_a[1])),(*reinterpret_cast< float*(*)>(_a[2])),(*reinterpret_cast< float(*)>(_a[3])),(*reinterpret_cast< float(*)>(_a[4]))); break;
        case 3: signalSouthernViewEntropyInitialized((*reinterpret_cast< ::size_t(*)>(_a[1])),(*reinterpret_cast< float*(*)>(_a[2])),(*reinterpret_cast< float(*)>(_a[3])),(*reinterpret_cast< float(*)>(_a[4]))); break;
        case 4: signalNorthernViewPointMarked((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 5: signalSouthernViewPointMarked((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 6: signalViewEntropyUpdated((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 7: signalLoadViewEntropy(); break;
        case 8: signalSaveViewEntropy(); break;
        case 9: slotUpdateTransferFunction((*reinterpret_cast< QHoverPoints*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< unsigned char(*)>(_a[3]))); break;
        case 10: slotUpdateVolumeStepSize((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 11: slotUpdateVolumeOffset((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 12: slotUpdateVolumeScale((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 13: slotUpdateComputingEntropyState((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 14: slotUpdateShadingState((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 15: slotUpdateGaussian1DState((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 16: slotUpdateGaussian2DState((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 17: slotUpdateLightPositionX((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 18: slotUpdateLightPositionY((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 19: slotUpdateLightPositionZ((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 20: slotUpdateLightColor((*reinterpret_cast< const QColor(*)>(_a[1]))); break;
        case 21: slotUpdateLightDiffuseCoeff((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 22: slotUpdateLightAmbientCoeff((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 23: slotUpdateLightSpecularCoeff((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 24: slotUpdateMaterialShininess((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 25: slotLoadConfigurations(); break;
        case 26: slotSaveConfigurations(); break;
        case 27: slotComputeViewEntropy(); break;
        case 28: slotMarkStartPoint(); break;
        case 29: slotMarkEndPoint(); break;
        default: ;
        }
        _id -= 30;
    }
    return _id;
}

// SIGNAL 0
void QVSWidget::signalHistogramInitialized(::size_t _t1, float * _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void QVSWidget::signalHistogramUpdated(unsigned int _t1, float * _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void QVSWidget::signalNorthernViewEntropyInitialized(::size_t _t1, float * _t2, float _t3, float _t4)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)), const_cast<void*>(reinterpret_cast<const void*>(&_t4)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void QVSWidget::signalSouthernViewEntropyInitialized(::size_t _t1, float * _t2, float _t3, float _t4)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)), const_cast<void*>(reinterpret_cast<const void*>(&_t4)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void QVSWidget::signalNorthernViewPointMarked(int _t1, int _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void QVSWidget::signalSouthernViewPointMarked(int _t1, int _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void QVSWidget::signalViewEntropyUpdated(double _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}

// SIGNAL 7
void QVSWidget::signalLoadViewEntropy()
{
    QMetaObject::activate(this, &staticMetaObject, 7, 0);
}

// SIGNAL 8
void QVSWidget::signalSaveViewEntropy()
{
    QMetaObject::activate(this, &staticMetaObject, 8, 0);
}
QT_END_MOC_NAMESPACE
