#include "neural_network.h"

#include <iostream>
#include <utility>
#include <valarray>

#include "layer.h"
#include "activation.h"

neural_network::neural_network(std::vector<int> sizes,
                               std::vector<std::function<void(std::vector<double>&)>> activations,
                               std::vector<std::function<void(std::vector<double>&)>> activations_derivatives) {
    for (size_t i = 1; i < sizes.size(); i++) {
        layers.emplace_back(sizes[i], sizes[i - 1], activations[i - 1], activations_derivatives[i - 1]);
    }
}

std::vector<double> neural_network::forward(std::vector<double> input) {
    std::vector<double> result = std::move(input);
    for (auto& layer : layers) {
        result = layer.forward(result);
    }
    return result;
}

std::vector<double> neural_network::logits(std::vector<double> input) {
    return forward(std::move(input));
}

int neural_network::predict(std::vector<double> input) {
    std::vector<double> result = forward(std::move(input));
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

void neural_network::backward(std::vector<double> predicted, std::vector<double> label, double learning_rate, std::vector<double> input) {
    std::vector<double> de_dy(predicted.size());
    matrix weight_delta = matrix(0, 0);

    for (int i = layers.size() - 1; i >= 0; i--) {
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
        y_vector->push_back(1);
        weight_delta = outer_product(de_dp,*y_vector);

        if (i == 0) {
            layer.weights_delta->substract(weight_delta * -1);
        }
    }
}

void neural_network::train(matrix inputs, std::vector<int> labels, int epochs, double learning_rate, bool debug_mode) {
    if (debug_mode) {
        std::cout << "BEFORE" << std::endl;
        visualize();
    }

    for (int i = 0; i < epochs; i++) {
        double error = 0;
        for (int j = 0; j < inputs.rows; j++) {
            std::vector<double> input = inputs.data[j];
            std::vector<double> label(layers.back().size, 0);
            label[labels[j]] = 1;

            std::vector<double> predicted = forward(input);
            softmax(predicted);
            for (size_t k = 0; k < predicted.size(); k++) {
                // cross entropy
                error += -label[k] * log(predicted[k]);
            }
        }

//        if (i % 100 == 0) {
//            std::cout << "Error at epoch " << i << ": " << error << std::endl;
//        }

        for (int j = 0; j < inputs.rows; j++) {
            if (j > 0 && j % 100 == 0) {
                std::cout << "Example number: " << j << std::endl;
            }
//            if (j > 10000) break;
            std::vector<double> input = inputs.data[j];
            std::vector<double> label(layers.back().size, 0);
            label[labels[j]] = 1;

            std::vector<double> predicted = forward(input);
            backward(predicted, label, learning_rate, input);

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
                for (auto &val: predicted) {
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