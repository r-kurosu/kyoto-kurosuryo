 #include <Python.h>
//#include "../../../include/python3.6m/Python.h"

// C function "get param and return list"
static PyObject* c_param_list(PyObject* self, PyObject* args)
{
    int a, b;
    PyObject* c_list;

    // Decide variable type (int, int)
    if (!PyArg_ParseTuple(args, "ii", &a, &b)){
        return NULL;
    }

    // multiplication
    a = a * 2;
    b = b * 2;

    // make python list	(length 2)
    c_list = PyList_New(2);

    // set param
    PyList_SET_ITEM(c_list, 0, PyLong_FromLong(a));
    PyList_SET_ITEM(c_list, 1, PyLong_FromLong(b));

    return c_list;
}

// Function Definition struct
static PyMethodDef plistMethods[] = {
    { "c_param_list", c_param_list, METH_VARARGS, "multiplication and make list"},
    { NULL }
};

// Module Definition struct
static struct PyModuleDef plistModule = {
    PyModuleDef_HEAD_INIT,
    "plistModule",
    "Python3 C API Module(Sample 3)",
    -1,
    plistMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_plistModule(void)
{
    return PyModule_Create(&plistModule);
}
