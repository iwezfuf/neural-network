#include "neural_network.h"

#include <iostream>
#include <random>
#include <numeric>
#include <fstream>

#include "layer.h"
#include "activation.h"

neural_network::neural_network(std::vector<int> sizes, const std::vector<Activation> &activations) {
    for (size_t i = 1; i < sizes.size(); i++) {
        layers.emplace_back(sizes[i], sizes[i - 1], activations[i - 1]);
    }
}

void neural_network::forward(const matrix_row_view &input) {
    layers[0].forward(input);
    for (size_t i = 1; i < layers.size(); i++) {
        layers[i].forward(*layers[i - 1].values);
    }
}

int neural_network::predict(const matrix_row_view& input) {
    forward(input);
    return static_cast<int>(std::max_element(get_outputs().begin(), get_outputs().end()) - get_outputs().begin());
}

int neural_network::accuracy(const matrix &inputs, const std::vector<int> &labels) {
    int correct = 0;
    for (int i = 0; i < inputs.rows; i++) {
        if (predict(inputs.get_row(i)) == labels[i]) {
            correct++;
        }
    }
    return correct * 100 / inputs.rows;
}

void neural_network::predict_to_file(const matrix &inputs, const std::string& filename) {
    std::string to_write;
    for (int i = 0; i < inputs.rows; i++) {
        to_write += std::to_string(predict(inputs.get_row(i))) + "\n";
    }
    std::ofstream file(filename);
    file << to_write;
    file.close();
}

void neural_network::backward(const matrix_row_view &input, const std::vector<float> &label) {
    std::vector<float> &predicted = get_outputs();
    std::vector<float> de_dy(predicted.size());
    matrix current_weights_delta = matrix(0, 0);

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
        auto& layer = layers[i];

        // compute de_dy
        if (static_cast<size_t>(i) == layers.size() - 1) {
            for (size_t j = 0; j < de_dy.size(); j++) {
                de_dy[j] = predicted[j] - label[j];
            }
        } else {
            de_dy = compute_de_dy(de_dy, *layers[i + 1].potential_der, *layers[i + 1].weights);
        }

        // compute de_dp = de_dy * potential (elementwise_mul)
        std::vector<float> de_dp = vec_elementwise_mul(de_dy, *layer.potential_der);

        if (static_cast<size_t>(i) != layers.size() - 1) {
            layers[i + 1].optimizer->add_current_example_weight_gradient(*layers[i + 1].weights_delta, current_weights_delta);
        }

        // compute de_dw
        const matrix_row_view *y_vector;
        if (i == 0) {
            y_vector = &input;
        } else {
            y_vector = new matrix_row_view(layers[i - 1].values->data(), layers[i - 1].size);
        }
        // outer product with 1 added to end of y_vector
        current_weights_delta = outer_product(de_dp, *y_vector);

        if (i == 0) {
            layer.optimizer->add_current_example_weight_gradient(*layer.weights_delta, current_weights_delta);
        }
    }
}

void neural_network::train(const matrix &inputs, const std::vector<int> &labels, int epochs, float learning_rate) {
    // sort inputs and labels randomly
    std::vector<int> indices(inputs.rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device("/dev/random")()));

    for (int i = 0; i < epochs; i++) {
        if (i % 100 == 0)
            std::cout << "Epoch: " << i << std::endl;

        int batch_size = std::min(32, inputs.rows);
        for (int j = 0; j < batch_size; j++) {
            int index = indices[(i*batch_size + j) % inputs.rows];

            const matrix_row_view &input = inputs.get_row(index);
            std::vector<float> label(layers.back().size, 0);
            label[labels[index]] = 1;

            forward(input);
            backward(input, label);
        }
        for (auto& layer : layers) {
            layer.update_weights(learning_rate);
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
                    std::cout << "Bias: " << layer.weights->data[layer.weights->cols * j + k] << " ";
                } else {
                    std::cout << "Weight " << j << " " << k << ": " << layer.weights->data[layer.weights->cols * j + k] << "         ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

std::vector<float> &neural_network::get_outputs() {
    return *layers[layers.size() - 1].values;
}
