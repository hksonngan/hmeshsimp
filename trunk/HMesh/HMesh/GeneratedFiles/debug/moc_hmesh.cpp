/****************************************************************************
** Meta object code from reading C++ file 'hmesh.h'
**
** Created: Fri Jul 27 10:30:10 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../hmesh.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'hmesh.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_HMesh[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
       7,    6,    6,    6, 0x0a,
      22,    6,    6,    6, 0x0a,
      33,    6,    6,    6, 0x0a,
      44,    6,    6,    6, 0x0a,
      59,    6,    6,    6, 0x0a,
      75,    6,    6,    6, 0x0a,
      85,    6,    6,    6, 0x0a,
     101,    6,    6,    6, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_HMesh[] = {
    "HMesh\0\0on_open_file()\0on_qslim()\0"
    "on_psimp()\0on_wireframe()\0on_flat_lines()\0"
    "on_flat()\0on_vert_color()\0on_face_color()\0"
};

const QMetaObject HMesh::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_HMesh,
      qt_meta_data_HMesh, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &HMesh::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *HMesh::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *HMesh::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_HMesh))
        return static_cast<void*>(const_cast< HMesh*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int HMesh::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: on_open_file(); break;
        case 1: on_qslim(); break;
        case 2: on_psimp(); break;
        case 3: on_wireframe(); break;
        case 4: on_flat_lines(); break;
        case 5: on_flat(); break;
        case 6: on_vert_color(); break;
        case 7: on_face_color(); break;
        default: ;
        }
        _id -= 8;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
