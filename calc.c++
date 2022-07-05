#include <pybind11/pybind11.h>

int calc_euclid(int x, int y) {
    double distance
    double sum = 0
    for(int i=0;i<N;i++){
        sum = sum + pow(x[i]-y[i])
    }
    distance = sqrt(sum)

    return distance;
}

PYBIND11_MODULE(calc, m) {
    m.def("calc", &calc_euclid);
}