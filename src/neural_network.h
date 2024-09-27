#ifndef NEURAL_NETWORK_NEURAL_NETWORK_H
#define NEURAL_NETWORK_NEURAL_NETWORK_H


#include "matrix.h"
#include "layer.h"

struct neural_network {
    std::vector<layer> layers;
public:
    neural_network(std::vector<int> sizes, std::vector<std::function<float(float)>> activations);
    void train(matrix inputs, std::vector<int> labels, int epochs, float learning_rate);
    int predict(std::vector<int> input);
    std::vector<float> forward(std::vector<int> input);
    void backward(std::vector<int> predicted, std::vector<int> label, float learning_rate);
};


#endif //NEURAL_NETWORK_NEURAL_NETWORK_H
