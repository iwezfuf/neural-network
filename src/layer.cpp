#include "layer.h"

#include <memory>

#include "matrix.h"

layer::layer(int size, int size_incoming, std::function<float(float)> activation) {
    this->size = size;
    this->size_incoming = size_incoming;
    this->activation = activation;
    this->weights = std::make_unique<matrix>(matrix(size_incoming, size));
}

std::vector<float> layer:: forward(std::vector<float> input) {
    auto result = (*weights) * input;
    vec_apply(result, this->activation);
    return result;
}

