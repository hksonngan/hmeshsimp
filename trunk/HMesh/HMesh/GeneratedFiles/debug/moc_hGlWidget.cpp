/****************************************************************************
** Meta object code from reading C++ file 'hGlWidget.h'
**
** Created: Thu Jan 31 12:19:31 2013
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../hGlWidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'hGlWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_hGlWidget[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      51,   11,   10,   10, 0x0a,
     115,   97,   92,   10, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_hGlWidget[] = {
    "hGlWidget\0\0filename,isovalue,deimateRate,maxNewTri\0"
    "setDrawMCSimp(string,double,double,uint)\0"
    "bool\0filename,isovalue\0setDrawMC(string,double)\0"
};

const QMetaObject hGlWidget::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_hGlWidget,
      qt_meta_data_hGlWidget, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &hGlWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *hGlWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *hGlWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_hGlWidget))
        return static_cast<void*>(const_cast< hGlWidget*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int hGlWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: setDrawMCSimp((*reinterpret_cast< string(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4]))); break;
        case 1: { bool _r = setDrawMC((*reinterpret_cast< string(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        default: ;
        }
        _id -= 2;
    }
    return _id;
}
QT_END_MOC_NAMESPACE