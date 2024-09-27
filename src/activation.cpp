#include "activation.h"
#include <cmath>

double relu(float x) {
    return x > 0 ? x : 0;
}

double relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

double sigmoid(float x) {
    return 1 / (1 + pow(M_E, -x));
}

double sigmoid_derivative(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> result;
    float sum = 0;
    for (float i : x) {
        sum += pow(M_E, i);
    }
    for (float i : x) {
        result.push_back(pow(M_E, i) / sum);
    }
    return result;
}
