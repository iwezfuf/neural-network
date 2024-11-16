#include "activation.h"
#include <cmath>
#include <cassert>

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
    for (float i : x) {
        sum += exp(i);
    }
    for (float & i : x) {
        i = exp(i) / sum;
    }
}

void softmax_derivative(std::vector<float> &x) {
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = 1;
    }
}

std::function<void(std::vector<float>&)> get_activation(Activation activation) {
    switch (activation) {
        case Activation::RELU:
            return relu;
        case Activation::SOFTMAX:
            return softmax;
    }
    assert(false);
}

std::function<void(std::vector<float>&)> get_activation_derivative(Activation activation) {
    switch (activation) {
        case Activation::RELU:
            return relu_derivative;
        case Activation::SOFTMAX:
            return softmax_derivative;
    }
    assert(false);
}
