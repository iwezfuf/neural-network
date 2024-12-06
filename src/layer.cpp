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
    if (has_nan(result)) {
        std::cout << "NaN weight found after calc_potentials" << std::endl;
        std::exit(42);
    }

    auto result_copy = result;
    get_activation_derivative(activation)(result_copy);
    if (has_nan(result_copy)) {
        std::cout << "NaN weight found after activation derivative" << std::endl;
        std::exit(42);
    }
    potential_der = std::make_unique<std::vector<float>>(result_copy);

    get_activation(activation)(result);
    if (has_nan(result)) {
        std::cout << "NaN weight found after activation" << std::endl;
        std::exit(42);
    }
    values = std::make_unique<std::vector<float>>(result);
}

void layer::update_weights(float learning_rate) const {
    optimizer->update_weights(*weights, *weights_delta, learning_rate);
}

bool layer::has_nan(std::vector<float> &vec) {
    for (auto &i : vec) {
        if (std::isnan(i)) {
            return true;
        }
    }
    return false;
}