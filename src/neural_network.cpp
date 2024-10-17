#include "neural_network.h"

#include <iostream>
#include <random>
#include <numeric>

#include "layer.h"
#include "activation.h"

neural_network::neural_network(std::vector<int> sizes,
                               std::vector<std::function<void(std::vector<double>&)>> activations,
                               std::vector<std::function<void(std::vector<double>&)>> activations_derivatives) {
    for (size_t i = 1; i < sizes.size(); i++) {
        layers.emplace_back(sizes[i], sizes[i - 1], activations[i - 1], activations_derivatives[i - 1]);
    }
}

void neural_network::forward(const std::vector<double> &input) {
    layers[0].forward(input);
    for (size_t i = 1; i < layers.size(); i++) {
        layers[i].forward(*layers[i - 1].values);
    }
}

std::vector<double> neural_network::logits(const std::vector<double> &input) {
    forward(input);
    return get_outputs();
}

int neural_network::predict(const std::vector<double>& input) {
    forward(input);
//    std::cout << "Result: " << std::endl;
//    for (auto& val : result) {
//        std::cout << val << " ";
//    }
//    std::cout << std::endl;
    int max_index = 0;
    auto& output = get_outputs();
    for (size_t i = 0; i < output.size(); i++) {
        if (output[i] > output[max_index]) {
            max_index = static_cast<int>(i);
        }
    }
    return max_index;
}

void neural_network::backward(std::vector<double> input, std::vector<double> label) {
    std::vector<double> &predicted = get_outputs();
    std::vector<double> de_dy(predicted.size());
    matrix weight_delta = matrix(0, 0);

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
        auto& layer = layers[i];

        // compute de_dy
        if (static_cast<size_t>(i) == layers.size() - 1) {
            for (size_t j = 0; j < de_dy.size(); j++) {
                de_dy[j] = predicted[j] - label[j];
            }
        } else {
            de_dy = (matrix({vec_elementwise_mul(de_dy, *layers[i + 1].potential_der)}) * layers[i + 1].weights->without_last_col()).data[0];
        }

        // compute de_dp - de_dy * potential (elementwise_mul)
        std::vector<double> de_dp = vec_elementwise_mul(de_dy, *layer.potential_der);

        // print de_dp
//        std::cout << "de_dp: " << " on layer " << i + 1 << std::endl;
//        for (auto& val : de_dp) {
//            std::cout << val << " ";
//        }
//        std::cout << std::endl;

        if (static_cast<size_t>(i) != layers.size() - 1) {
            layers[i + 1].weights_delta->substract(weight_delta * -1);
        }

        // compute de_dw
        std::vector<double> *y_vector;
        if (i == 0) {
            y_vector = &input;
        } else {
            y_vector = layers[i - 1].values.get();
        }
        // outer product with 1 added to end of y_vector
        weight_delta = outer_product(de_dp,*y_vector);

        if (i == 0) {
            layer.weights_delta->substract(weight_delta * -1);
        }
    }
}

void neural_network::train(const matrix &inputs, const std::vector<int> &labels, int epochs, double learning_rate, bool debug_mode) {
    if (debug_mode) {
        std::cout << "BEFORE" << std::endl;
        visualize();
    }

    // sort inputs and labels randomly
    std::vector<int> indices(inputs.rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

    for (int i = 0; i < epochs; i++) {
        int batch_size = std::min(20, inputs.rows);
        for (int j = 0; j < batch_size; j++) {
            int index = indices[(i*batch_size + j) % inputs.rows];

            if (j > 0 && j % 100 == 0) {
                std::cout << "Example number: " << j << std::endl;
            }
//            if (j > 10000) break;
            const std::vector<double> &input = inputs.data[index];
            std::vector<double> label(layers.back().size, 0);
            label[labels[index]] = 1;

            forward(input);
            backward(input, label);

            // print weight gradients
//            std::cout << "Input: " << input[0] << " " << input[1] << " Logit for 0: " << logits(inputs.data[j])[0] << " Logit for 1: " << logits(inputs.data[j])[1];
//            std::cout << std::endl;
//            for (auto& layer : layers) {
//                for (int k = 0; k < layer.weights_delta->rows; k++) {
//                    for (int l = 0; l < layer.weights_delta->cols; l++) {
//                        if (l == layer.weights_delta->cols - 1) {
//                            std::cout << "Bias: " << layer.weights_delta->data[k][l] << " ";
//                        } else {
//                            std::cout << "Weight " << k << " " << l << ": " << layer.weights_delta->data[k][l]
//                                      << "         ";
//                        }
//                        std::cout << std::endl;
//                    }
//                }
//            }


            if (debug_mode) {
                std::cout << "AFTER epoch: " << i << " Example: [ ";
                for (auto &val: input) {
                    std::cout << val << " ";
                }
                std::cout << "] Predicted: [ ";
                for (auto &val: *layers[layers.size() - 1].values) {
                    std::cout << val << " ";
                }
                std::cout << "] Label: [ ";
                for (auto &val: label) {
                    std::cout << val << " ";
                }
                std::cout << "]" << std::endl;
                visualize();
                std::cout << std::endl;
            }
//            break;
        }
        for (auto& layer : layers) {
//            // print weight gradients
//            for (int k = 0; k < layer.weights_delta->rows; k++) {
//                for (int l = 0; l < layer.weights_delta->cols; l++) {
//                    if (l == layer.weights_delta->cols - 1) {
//                        std::cout << "Bias: " << layer.weights_delta->data[k][l] << " ";
//                    } else {
//                        std::cout << "Weight " << k << " " << l << ": " << layer.weights_delta->data[k][l]
//                                  << "         ";
//                    }
//                    std::cout << std::endl;
//                }
//            }
            layer.update_weights(learning_rate);
            layer.zero_weights_delta();
        }
    }
}

void neural_network::visualize() {
    for (size_t i = 0; i < layers.size(); i++) {
        auto& layer = layers[i];
        std::cout << "Layer: " << i << std::endl;
        for (int j = 0; j < layer.weights->rows; j++) {
            for (int k = 0; k < layer.weights->cols; k++) {
                if (k == layer.weights->cols - 1) {
                    std::cout << "Bias: " << layer.weights->data[j][k] << " ";
                } else {
                    std::cout << "Weight " << j << " " << k << ": " << layer.weights->data[j][k] << "         ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

std::vector<double> &neural_network::get_outputs() {
    return *layers[layers.size() - 1].values;
}
