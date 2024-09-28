#ifndef NEURAL_NETWORK_ACTIVATION_H
#define NEURAL_NETWORK_ACTIVATION_H

#include <vector>

void relu(std::vector<float>& x);
void relu_derivative(std::vector<float>& x);

void sigmoid(const std::vector<float>& x);
void sigmoid_derivative(std::vector<float>& x);;

void softmax(std::vector<float>& x);
void softmax_derivative(std::vector<float>& x);


#endif //NEURAL_NETWORK_ACTIVATION_H
