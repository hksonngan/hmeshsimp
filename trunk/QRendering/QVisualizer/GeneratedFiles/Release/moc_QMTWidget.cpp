/****************************************************************************
** Meta object code from reading C++ file 'QMTWidget.h'
**
** Created: Thu Dec 13 14:28:37 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../MarchingTetrahedrons/QMTWidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QMTWidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QMTWidget[] = {

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

static const char qt_meta_stringdata_QMTWidget[] = {
    "QMTWidget\0\0unsigned char\0slotUpdateIsoValue()\0"
};

const QMetaObject QMTWidget::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_QMTWidget,
      qt_meta_data_QMTWidget, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QMTWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QMTWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QMTWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QMTWidget))
        return static_cast<void*>(const_cast< QMTWidget*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int QMTWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
