/****************************************************************************
** Meta object code from reading C++ file 'QTransferFunction1D.h'
**
** Created: Wed Jan 2 15:03:26 2013
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../infrastructures/QTransferFunction1D.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QTransferFunction1D.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QTransferFunction1D[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      50,   21,   20,   20, 0x05,

 // slots: signature, parameters, type, tag, flags
     138,  110,   20,   20, 0x0a,
     201,  175,   20,   20, 0x0a,
     243,  234,   20,   20, 0x0a,
     290,  285,   20,   20, 0x0a,
     326,   20,   20,   20, 0x2a,
     351,  285,   20,   20, 0x0a,
     387,   20,   20,   20, 0x2a,

       0        // eod
};

static const char qt_meta_stringdata_QTransferFunction1D[] = {
    "QTransferFunction1D\0\0controlPoints,width,modified\0"
    "signalControlPointsChanged(QHoverPoints*,int,unsigned char)\0"
    "histogramSize,histogramData\0"
    "slotInsertHistogram(::size_t,float*)\0"
    "histogramID,histogramData\0"
    "slotUpdateHistogram(uint,float*)\0"
    "modified\0slotUpdateTransferFunction(unsigned char)\0"
    "name\0slotLoadConfigurations(std::string)\0"
    "slotLoadConfigurations()\0"
    "slotSaveConfigurations(std::string)\0"
    "slotSaveConfigurations()\0"
};

const QMetaObject QTransferFunction1D::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_QTransferFunction1D,
      qt_meta_data_QTransferFunction1D, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QTransferFunction1D::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QTransferFunction1D::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QTransferFunction1D::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QTransferFunction1D))
        return static_cast<void*>(const_cast< QTransferFunction1D*>(this));
    return QWidget::qt_metacast(_clname);
}

int QTransferFunction1D::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: signalControlPointsChanged((*reinterpret_cast< QHoverPoints*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< unsigned char(*)>(_a[3]))); break;
        case 1: slotInsertHistogram((*reinterpret_cast< ::size_t(*)>(_a[1])),(*reinterpret_cast< float*(*)>(_a[2]))); break;
        case 2: slotUpdateHistogram((*reinterpret_cast< uint(*)>(_a[1])),(*reinterpret_cast< float*(*)>(_a[2]))); break;
        case 3: slotUpdateTransferFunction((*reinterpret_cast< unsigned char(*)>(_a[1]))); break;
        case 4: slotLoadConfigurations((*reinterpret_cast< std::string(*)>(_a[1]))); break;
        case 5: slotLoadConfigurations(); break;
        case 6: slotSaveConfigurations((*reinterpret_cast< std::string(*)>(_a[1]))); break;
        case 7: slotSaveConfigurations(); break;
        default: ;
        }
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void QTransferFunction1D::signalControlPointsChanged(QHoverPoints * _t1, int _t2, unsigned char _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
