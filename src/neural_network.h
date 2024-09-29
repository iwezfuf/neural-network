#ifndef NEURAL_NETWORK_NEURAL_NETWORK_H
#define NEURAL_NETWORK_NEURAL_NETWORK_H

#include <vector>
#include <functional>

#include "matrix.h"
#include "layer.h"

struct neural_network {
    std::vector<layer> layers;
    std::function<double(std::vector<double>, std::vector<double>)> error_function;
public:
    neural_network(std::vector<int> sizes,
                   std::vector<std::function<void(std::vector<double>&)>> activations,
                   std::vector<std::function<void(std::vector<double>&)>> activation_derivatives);

    void train(matrix inputs, std::vector<int> labels, int epochs, double learning_rate);

    int predict(std::vector<double> input);

    std::vector<double> forward(std::vector<double> input);

    void backward(std::vector<double> predicted, std::vector<double> label, double learning_rate, std::vector<double> input);

    void visualize();
};


#endif //NEURAL_NETWORK_NEURAL_NETWORK_H
