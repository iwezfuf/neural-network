#include "layer.h"

#include <memory>
#include <vector>

#include "matrix.h"
#include "activation.h"

layer::layer(int size, int size_incoming, Activation activation) {
    this->size = size;
    this->activation = activation;
    this->weights = std::make_unique<matrix>(matrix(size, size_incoming + 1));
    this->weights->randomize();
    this->weights_delta = std::make_unique<matrix>(matrix(size, size_incoming + 1));
    this->values = std::make_unique<std::vector<double>>(std::vector<double>(size));
}

void layer::forward(const matrix_row_view &input) {
    std::vector<double> result = weights->calc_potentials(input);

    auto result_copy = result;
    get_activation_derivative(activation)(result);
    potential_der = std::make_unique<std::vector<double>>(result_copy);

    get_activation(activation)(result);
    values = std::make_unique<std::vector<double>>(result);
}

void layer::update_weights(double learning_rate) const {
    *weights -= *weights_delta * learning_rate;
}

void layer::zero_weights_delta() const {
    weights_delta->zero();
}
