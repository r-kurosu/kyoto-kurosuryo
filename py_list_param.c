 #include <Python.h>
//#include "../../../include/python3.6m/Python.h"
 #include <math.h>

// C function "get list and return param"
static PyObject* c_list_param(PyObject* self, PyObject* args)
{
    int n;
//    long a, b, sum = 0;
    double a, b;
    double sum = 0.0;
    PyObject* x_list, *item_x, *item_y;
//    double *item_y;

    // Decide variable type (list)
    if (!PyArg_ParseTuple(args, "O", &x_list)){
        return NULL;
    }

    // Check list
    if (PyList_Check(x_list)){
        // get length of the list
        n = PyList_Size(x_list);
    }else{
        return NULL;
    }

    // Calculate list sum -> calculate euclid dist
    for (int i = 0; i < n/2; i++){
        item_x = PyList_GetItem(x_list, i);
        a = PyFloat_AsDouble(item_x); // Increment the reference count

        item_y = PyList_GetItem(x_list, i + n/2);
        b = PyFloat_AsDouble(item_y); // Increment the reference count

        sum = sum + pow((a-b), 2.0);

        Py_DECREF(item_x); // Decrement the reference count
        Py_DECREF(item_y); // Decrement the reference count
    }

    Py_DECREF(x_list); // Decrement the reference count
//    Py_DECREF(y_list); // Decrement the reference count

    return Py_BuildValue("d", sum);
}

// Function Definition struct
static PyMethodDef listpMethods[] = {
    { "sum_list", c_list_param, METH_VARARGS, "Calculate list sum"},
    { NULL }
};

// Module Definition struct
static struct PyModuleDef listpModule = {
    PyModuleDef_HEAD_INIT,
    "listpModule",
    "Python3 C API Module(Sample 4)",
    -1,
    listpMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_listpModule(void)
{
    return PyModule_Create(&listpModule);
}
