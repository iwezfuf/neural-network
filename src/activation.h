#ifndef NEURAL_NETWORK_ACTIVATION_H
#define NEURAL_NETWORK_ACTIVATION_H

#include <vector>

double relu(float x);
double relu_derivative(float x);

double sigmoid(float x);
double sigmoid_derivative(float x);

std::vector<float> softmax(const std::vector<float>& x);
std::vector<float> softmax_derivative(const std::vector<float>& x);


#endif //NEURAL_NETWORK_ACTIVATION_H
