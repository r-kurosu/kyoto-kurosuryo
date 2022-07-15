 #include <Python.h>
//#include "../../../include/python3.6m/Python.h"
 #include <math.h>
// #include <stdio.h>

// C function "get list and return param"
static PyObject* c_list_param(PyObject* self, PyObject* args)
{
//    printf("Hello world\n");

    int n;
    double a, b;
    double sum = 0.0;
    PyObject* c_list, *item_x, *item_y;
//    double *item_y;

    // Decide variable type (list)
    if (!PyArg_ParseTuple(args, "O", &c_list)){
        return NULL;
    }
    Py_INCREF(c_list); // Decrement the reference count

    // Check list
    if (PyList_Check(c_list)){
        // get length of the list
        n = PyList_Size(c_list);
    }else{
        return NULL;
    }
//    printf("Hello world\n");

    // Calculate list sum -> calculate euclid dist
    for (int i = 0; i < n/2; i++){
        item_x = PyList_GetItem(c_list, i);
        Py_INCREF(item_x);
        a = PyFloat_AsDouble(item_x); // Increment the reference count

        item_y = PyList_GetItem(c_list, i + n/2);
        Py_INCREF(item_y); // Decrement the reference count
        b = PyFloat_AsDouble(item_y); // Increment the reference count

//        sum = sum + pow((a-b), 2.0);
        sum = sum + sqrt(pow((a-b), 2.0));
//        sum = sum + (a-b)*(a-b);
//        sum = sum + sqrt((a-b)*(a-b));
//        printf("%d\n", i);
//        printf("Hello world\n");

        Py_DECREF(item_x); // Decrement the reference count
        Py_DECREF(item_y); // Decrement the reference count
    }

    Py_DECREF(c_list); // Decrement the reference count



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
