#include "neural_network.h"

#include "layer.h"

neural_network::neural_network(std::vector<int> sizes, std::vector<std::function<float(float)>> activations) {
    for (size_t i = 0; i < sizes.size() - 1; i++) {
        layers.push_back(layer(sizes[i], sizes[i + 1], activations[i]));
    }
}

std::vector<float> neural_network:: forward(std::vector<float> input) {
    std::vector<float> result = input;
    for (auto& layer : layers) {
        result = layer.forward(result);
    }
    return result;
}

int neural_network:: predict(std::vector<float> input) {
    std::vector<float> result = forward(input);
    int max_index = 0;
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] > result[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

void neural_network::backward(std::vector<float> predicted, std::vector<float> label, float learning_rate) {
}
