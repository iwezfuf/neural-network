#ifndef NEURAL_NETWORK_NEURAL_NETWORK_H
#define NEURAL_NETWORK_NEURAL_NETWORK_H

#include <vector>
#include <functional>

#include "matrix.h"
#include "layer.h"

struct neural_network {
    std::vector<layer> layers;
    std::function<float(std::vector<float>, std::vector<float>)> error_function;
public:
    neural_network(std::vector<int> sizes, std::vector<std::function<float(float)>> activations);

    void train(matrix inputs, std::vector<int> labels, int epochs, float learning_rate);

    int predict(std::vector<float> input);

    std::vector<float> forward(std::vector<float> input);

    void backward(std::vector<float> predicted, std::vector<float> label, float learning_rate);

};


#endif //NEURAL_NETWORK_NEURAL_NETWORK_H
