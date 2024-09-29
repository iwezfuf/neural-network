#include "neural_network.h"

#include <iostream>

#include "layer.h"

neural_network::neural_network(std::vector<int> sizes,
                               std::vector<std::function<void(std::vector<float>&)>> activations,
                               std::vector<std::function<void(std::vector<float>&)>> activations_derivatives) {
    for (size_t i = 1; i < sizes.size(); i++) {
        layers.push_back(layer(sizes[i], sizes[i - 1], activations[i - 1], activations_derivatives[i - 1]));
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
//    std::cout << "Result: " << std::endl;
//    for (auto& val : result) {
//        std::cout << val << " ";
//    }
//    std::cout << std::endl;
    int max_index = 0;
    for (size_t i = 0; i < result.size(); i++) {
        if (result[i] > result[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

void neural_network::backward(std::vector<float> predicted, std::vector<float> label, float learning_rate, std::vector<float> input) {
    std::vector<float> de_dy(predicted.size());

    for (int i = layers.size() - 1; i >= 0; i--) {
        auto& layer = layers[i];

        // compute de_dy
        if (static_cast<size_t>(i) == layers.size() - 1) {
            for (size_t j = 0; j < de_dy.size(); j++) {
                de_dy[j] = predicted[j] - label[j];
            }
        } else {
            layer.activation_derivative(*layers[i+1].potential);

            auto second = *layers[i+1].potential;
            auto mulled = vec_elementwise_mul(de_dy, second);
            auto third = layer.weights->without_last_col();
            auto result = (matrix({mulled}) * layers[i + 1].weights->without_last_col());

            de_dy = (matrix({vec_elementwise_mul(de_dy, *layers[i + 1].potential)}) * layers[i + 1].weights->without_last_col()).data[0];
        }

        // compute de_dp - dedy * potential (elementwise_mul)
        std::vector<float> de_dp = vec_elementwise_mul(de_dy, *layer.potential);
        // compute de_dw
        for (int j = 0; j < layer.weights->rows; j++) {
            for (int k = 0; k < layer.weights->cols; k++) {
                if (k == layer.weights->cols - 1) {
                    layer.weights->data[j][k] -= learning_rate * de_dp[j];
                } else {
                    if (i == 0) {
                        layer.weights->data[j][k] -= learning_rate * de_dp[j] * input[k];
                    } else {
                        layer.weights->data[j][k] -= learning_rate * de_dp[j] * layers[i - 1].values->at(k);
                    }
                }
            }
        }
    }
}

void neural_network::train(matrix inputs, std::vector<int> labels, int epochs, float learning_rate) {
    std::cout << "BEFORE" << std::endl;
    visualize();
    std::cout << "BEFORE" << std::endl;

    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < inputs.rows; j++) {
            std::vector<float> input = inputs.data[j];
            std::vector<float> label(layers.back().size, 0);
            label[labels[j]] = 1;

            std::vector<float> predicted = forward(input);
            backward(predicted, label, learning_rate, input);
        }
        std::cout << "Epoch: " << i << " Loss: " << std::endl;
        visualize();
    }
}

void neural_network::visualize() {
    for (auto& layer : layers) {
        for (int i = 0; i < layer.weights->rows; i++) {
            for (int j = 0; j < layer.weights->cols; j++) {
                std::cout << layer.weights->data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
}