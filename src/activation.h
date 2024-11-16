#ifndef NEURAL_NETWORK_ACTIVATION_H
#define NEURAL_NETWORK_ACTIVATION_H

#include <vector>
#include <functional>

void relu(std::vector<float>& x);
void relu_derivative(std::vector<float>& x);

void softmax(std::vector<float>& x);
void softmax_derivative(std::vector<float>& x);

enum class Activation {
    RELU,
    SOFTMAX
};

std::function<void(std::vector<float>&)> get_activation(Activation activation);

std::function<void(std::vector<float>&)> get_activation_derivative(Activation activation);

#endif //NEURAL_NETWORK_ACTIVATION_H
