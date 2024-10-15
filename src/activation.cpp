#include "activation.h"
#include <cmath>

#include "matrix.h"

double relu_single(double x) {
    return x > 0 ? x : 0;
}

void relu(std::vector<double>& x) {
    vec_apply(x, relu_single);
}

double relu_derivative_single(double x) {
    return x > 0 ? 1 : 0;
}

void relu_derivative(std::vector<double>& x) {
    vec_apply(x, relu_derivative_single);
}

void softmax(std::vector<double>& x) {
    double sum = 0;
    for (double i : x) {
        sum += exp(i);
    }
    for (double & i : x) {
        i = exp(i) / sum;
    }
}

void softmax_derivative(std::vector<double> &x) {
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = 1;
    }
}
