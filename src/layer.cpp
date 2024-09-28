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
    std::cout << "input: " << std::endl;
    for (auto& val : input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "weights: " << std::endl;
    for (int i = 0; i < weights->rows; i++) {
        for (int j = 0; j < weights->cols; j++) {
            std::cout << weights->data[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<float> result = (*weights) * input;
    values = std::make_unique<std::vector<float>>(result);
    activation(result);
    potential = std::make_unique<std::vector<float>>(result);
    return result;
}
