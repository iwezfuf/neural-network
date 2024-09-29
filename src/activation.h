#ifndef NEURAL_NETWORK_ACTIVATION_H
#define NEURAL_NETWORK_ACTIVATION_H

#include <vector>

void relu(std::vector<double>& x);
void relu_derivative(std::vector<double>& x);

void sigmoid(const std::vector<double>& x);
void sigmoid_derivative(std::vector<double>& x);;

void softmax(std::vector<double>& x);
void softmax_derivative(std::vector<double>& x);


#endif //NEURAL_NETWORK_ACTIVATION_H
