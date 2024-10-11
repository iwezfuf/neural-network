#include "layer.h"

#include <memory>
#include <functional>
#include <vector>
#include <iostream>

#include "matrix.h"

layer::layer(int size, int size_incoming, std::function<void(std::vector<double>&)> activation,
             std::function<void(std::vector<double>&)> activation_derivative) {
    this->size = size;
    this->size_incoming = size_incoming;
    this->activation = activation;
    this->activation_derivative = activation_derivative;
    this->weights = std::make_unique<matrix>(matrix(size, size_incoming + 1));
    this->weights->randomize();
    this->weights_delta = std::make_unique<matrix>(matrix(size, size_incoming + 1));
    this->values = std::make_unique<std::vector<double>>(std::vector<double>(size));
}

std::vector<double> layer::forward(std::vector<double> input) {
    input.push_back(1);
    std::vector<double> result = (*weights) * input;

    auto result_copy = result;
    this->activation_derivative(result_copy);
    potential_der = std::make_unique<std::vector<double>>(result_copy);

    activation(result);
    values = std::make_unique<std::vector<double>>(result);
    return result;
}

void layer::update_weights(double learning_rate) const {
    *weights -= *weights_delta * learning_rate;
}

void layer::zero_weights_delta() const {
    weights_delta->zero();
}
