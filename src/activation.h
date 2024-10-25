#ifndef NEURAL_NETWORK_ACTIVATION_H
#define NEURAL_NETWORK_ACTIVATION_H

#include <vector>
#include <functional>

void relu(std::vector<double>& x);
void relu_derivative(std::vector<double>& x);

void softmax(std::vector<double>& x);
void softmax_derivative(std::vector<double>& x);

enum class Activation {
    RELU,
    SOFTMAX
};

std::function<void(std::vector<double>&)> get_activation(Activation activation);

std::function<void(std::vector<double>&)> get_activation_derivative(Activation activation);

#endif //NEURAL_NETWORK_ACTIVATION_H
