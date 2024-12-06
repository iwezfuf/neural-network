#include "layer.h"

#include <memory>
#include <vector>

#include "matrix.h"
#include "activation.h"
#include "optimizers.h"

layer::layer(int size, int size_incoming, Activation activation) {
    this->size = size;
    this->activation = activation;
    this->weights = std::make_unique<matrix>(matrix(size, size_incoming + 1));
    this->weights->randomize();
    this->weights_delta = std::make_unique<matrix>(matrix(size, size_incoming + 1));
    this->values = std::make_unique<std::vector<float>>(std::vector<float>(size));
    this->optimizer = std::make_unique<adam_optimizer>(adam_optimizer(size, size_incoming + 1));
}

void layer::forward(const matrix_row_view &input) {
    std::vector<float> result = weights->calc_potentials(input);

    auto result_copy = result;
    get_activation_derivative(activation)(result_copy);
    potential_der = std::make_unique<std::vector<float>>(result_copy);

    get_activation(activation)(result);
    values = std::make_unique<std::vector<float>>(result);
}

void layer::update_weights(float learning_rate) const {
    optimizer->update_weights(*weights, *weights_delta, learning_rate);
}
