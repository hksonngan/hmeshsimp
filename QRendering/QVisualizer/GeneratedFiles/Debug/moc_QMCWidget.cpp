/****************************************************************************
** Meta object code from reading C++ file 'QMCWidget.h'
**
** Created: Tue Apr 17 11:29:36 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../MarchingCubes/QMCWidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QMCWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QMCWidget[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      25,   10,   11,   10, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_QMCWidget[] = {
    "QMCWidget\0\0unsigned char\0slotUpdateIsoValue()\0"
};

const QMetaObject QMCWidget::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_QMCWidget,
      qt_meta_data_QMCWidget, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QMCWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QMCWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QMCWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QMCWidget))
        return static_cast<void*>(const_cast< QMCWidget*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int QMCWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: { unsigned char _r = slotUpdateIsoValue();
            if (_a[0]) *reinterpret_cast< unsigned char*>(_a[0]) = _r; }  break;
        default: ;
        }
        _id -= 1;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
