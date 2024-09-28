#include "activation.h"
#include <cmath>

#include "matrix.h"

float relu_single(float x) {
    return x > 0 ? x : 0;
}

void relu(std::vector<float>& x) {
    vec_apply(x, relu_single);
}

float relu_derivative_single(float x) {
    return x > 0 ? 1 : 0;
}

void relu_derivative(std::vector<float>& x) {
    vec_apply(x, relu_derivative_single);
}

void softmax(std::vector<float>& x) {
    float sum = 0;
    for (size_t i = 0; i < x.size(); i++) {
        sum += exp(x[i]);
    }
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = exp(x[i]) / sum;
    }
}

void softmax_derivative(std::vector<float>&) {
    // TODO
}
