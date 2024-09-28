#include "layer.h"

#include <memory>
#include <functional>
#include <vector>
#include <iostream>

#include "matrix.h"

layer::layer(int size, int size_incoming, std::function<void(std::vector<float>&)> activation,
             std::function<void(std::vector<float>&)> activation_derivative) {
    this->size = size;
    this->size_incoming = size_incoming;
    this->activation = activation;
    this->activation_derivative = activation_derivative;
    this->weights = std::make_unique<matrix>(matrix(size, size_incoming + 1));
    this->weights->randomize();
    this->values = std::make_unique<std::vector<float>>(std::vector<float>(size));
}

std::vector<float> layer:: forward(std::vector<float> input) {
    input.push_back(1);
    std::vector<float> result = (*weights) * input;
    values = std::make_unique<std::vector<float>>(result);
    activation(result);
    potential = std::make_unique<std::vector<float>>(result);
    return result;
}
