/****************************************************************************
** Meta object code from reading C++ file 'QVisualizer.h'
**
** Created: Thu Dec 13 14:28:36 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../QVisualizer.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QVisualizer.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QVisualizer[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      13,   12,   12,   12, 0x08,
      36,   12,   12,   12, 0x08,
      59,   12,   12,   12, 0x08,
      82,   12,   12,   12, 0x08,
     105,   12,   12,   12, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_QVisualizer[] = {
    "QVisualizer\0\0slotInitVRVisualizer()\0"
    "slotInitMCVisualizer()\0slotInitMTVisualizer()\0"
    "slotInitDPVisualizer()\0slotInitVSVisualizer()\0"
};

const QMetaObject QVisualizer::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_QVisualizer,
      qt_meta_data_QVisualizer, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QVisualizer::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QVisualizer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QVisualizer::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QVisualizer))
        return static_cast<void*>(const_cast< QVisualizer*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int QVisualizer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: slotInitVRVisualizer(); break;
        case 1: slotInitMCVisualizer(); break;
        case 2: slotInitMTVisualizer(); break;
        case 3: slotInitDPVisualizer(); break;
        case 4: slotInitVSVisualizer(); break;
        default: ;
        }
        _id -= 5;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
