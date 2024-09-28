#include <iostream>
#include <vector>
#include "neural_network.h"
#include "matrix.h"
#include "activation.h"

int main() {
    auto *nn = new neural_network({8, 8, 2}, {relu, relu, softmax}, {relu_derivative, relu_derivative, softmax_derivative});

    matrix inputs(4, 2);
    inputs.data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    std::vector<int> labels = {0, 1, 1, 0};

    nn->train(inputs, labels, 2, 0.1);

    std::cout << nn->predict({0, 0}) << std::endl;
    std::cout << nn->predict({0, 1}) << std::endl;
    std::cout << nn->predict({1, 0}) << std::endl;
    std::cout << nn->predict({1, 1}) << std::endl;
    return 0;
}
