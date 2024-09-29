#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include "neural_network.h"
#include "matrix.h"
#include "activation.h"

void test_and() {
    auto *nn = new neural_network({2, 1, 2}, {relu, softmax}, {relu_derivative, softmax_derivative});
    matrix inputs(4, 2);
    inputs.data = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    std::vector<int> labels = {0, 0, 0, 1};
    nn->train(inputs, labels, 100, 0.1);

    std::vector<int> predicted = {nn->predict({0, 0}), nn->predict({1, 0}), nn->predict({0, 1}), nn->predict({1, 1})};
    if (predicted[0] == 0 && predicted[1] == 0 && predicted[2] == 0 && predicted[3] == 1) {
        std::cout << "AND test passed" << std::endl;
    } else {
        std::cout << "AND test failed" << std::endl;
    }
}

void test_or() {
    auto *nn = new neural_network({2, 1, 2}, {relu, softmax}, {relu_derivative, softmax_derivative});
    matrix inputs(4, 2);
    inputs.data = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    std::vector<int> labels = {0, 1, 1, 1};
    nn->train(inputs, labels, 100, 0.1);

    std::vector<int> predicted = {nn->predict({0, 0}), nn->predict({1, 0}), nn->predict({0, 1}), nn->predict({1, 1})};
    if (predicted[0] == 0 && predicted[1] == 1 && predicted[2] == 1 && predicted[3] == 1) {
        std::cout << "OR test passed" << std::endl;
    } else {
        std::cout << "OR test failed" << std::endl;
    }
}

void test_xor() {
    auto *nn = new neural_network({2, 2, 2, 2}, {relu, relu, softmax}, {relu_derivative, relu_derivative, softmax_derivative});
    matrix inputs(4, 2);
    inputs.data = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    std::vector<int> labels = {0, 1, 1, 0};
    nn->train(inputs, labels, 300, 0.05);

    std::vector<int> predicted = {nn->predict({0, 0}), nn->predict({1, 0}), nn->predict({0, 1}), nn->predict({1, 1})};
    if (predicted[0] == 0 && predicted[1] == 1 && predicted[2] == 1 && predicted[3] == 0) {
        std::cout << "XOR test passed" << std::endl;
    } else {
        std::cout << "XOR test failed" << std::endl;
    }
}

void test_larger() {
    auto *nn = new neural_network({2, 2}, {softmax}, {softmax_derivative});
    matrix inputs(7, 2);
    inputs.data = {{0, 1}, {1, 0}, {2, 3}, {3, 2}, {1, 2}, {2, 1}, {100, 10}};
    std::vector<int> labels = {0, 1, 0, 1, 0, 1, 1};
    nn->train(inputs, labels, 100, 0.1);

    std::vector<int> predicted = {nn->predict({0, 1}), nn->predict({1, 0}), nn->predict({2, 3}), nn->predict({3, 2}), nn->predict({1, 2}), nn->predict({2, 1}), nn->predict({100, 10})};
    if (predicted[0] == 0 && predicted[1] == 1 && predicted[2] == 0 && predicted[3] == 1 && predicted[4] == 0 && predicted[5] == 1 && predicted[6] == 1) {
        std::cout << "Larger test passed" << std::endl;
    } else {
        std::cout << "Larger test failed, outputs: " << std::endl;
        for (auto& val : predicted) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void dataset() {
    const std::string labels_file = "data/fashion_mnist_train_labels.csv";
    const std::string vectors_file = "data/fashion_mnist_train_vectors.csv";

    std::vector<int> labels;
    std::vector<std::vector<double>> vectors;

    std::ifstream labels_stream(labels_file);
    if (!labels_stream.is_open()) {
        std::cerr << "Failed to open labels file: " << labels_file << std::endl;
        return;
    }

    int label;
    while (labels_stream >> label) {
        labels.push_back(label);
    }
    labels_stream.close();

    std::ifstream vectors_stream(vectors_file);
    if (!vectors_stream.is_open()) {
        std::cerr << "Failed to open vectors file: " << vectors_file << std::endl;
        return;
    }

    std::string line;
    while (std::getline(vectors_stream, line)) {
        std::vector<double> vector;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            vector.push_back(std::stod(value));
        }
        vectors.push_back(vector);
    }
    vectors_stream.close();

    std::cout << "Loaded " << labels.size() << " labels and " << vectors.size() << " vectors." << std::endl;
    std::cout << "First label: " << labels[0] << std::endl;
    std::cout << "First vector: ";
    for (auto& val : vectors[0]) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Normalize vectors
    for (auto& vector : vectors) {
        double sum = 0;
        for (auto& val : vector) {
            sum += val;
        }
        for (auto& val : vector) {
            val /= sum;
        }
    }

    auto *nn = new neural_network({784, 521, 128, 10}, {relu, relu, softmax}, {relu_derivative, relu_derivative, softmax_derivative});
    matrix inputs(vectors.size(), 784);
    for (size_t i = 0; i < vectors.size(); i++) {
        for (size_t j = 0; j < vectors[i].size(); j++) {
            inputs.data[i][j] = vectors[i][j];
        }
    }
    nn->train(inputs, labels, 1, 0.1);

    std::vector<int> predicted;
    for (size_t i = 0; i < vectors.size(); i++) {
        predicted.push_back(nn->predict(vectors[i]));
    }

    // print accuracy
    int correct = 0;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] == predicted[i]) {
            correct++;
        }
    }
    std::cout << "Accuracy: " << static_cast<float>(correct) / labels.size() << std::endl;
}

int main() {
    test_and();
    test_or();
    test_larger();
    test_xor();

//    auto *nn = new neural_network({2, 1, 2}, {relu, softmax}, {relu_derivative, softmax_derivative});
//
//    matrix inputs(4, 2);
//    inputs.data = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
//
//    std::vector<int> labels = {0, 1, 1, 0};
//
//    nn->train(inputs, labels, 500, 0.05);
//
//    std::vector<int> predicted = {nn->predict({0, 0}), nn->predict({1, 0}), nn->predict({0, 1}), nn->predict({1, 1})};
//    for (auto& val : predicted) {
//        std::cout << val << " ";
//    }

    dataset();
    return 0;
}

//int main() {
//    auto *nn = new neural_network({1, 1, 2}, {relu, softmax}, {relu_derivative, softmax_derivative});
//
//    matrix inputs(2, 1);
//    inputs.data = {{0}, {1}};
//
//    std::vector<int> labels = {0, 1};
//
//    nn->train(inputs, labels, 100, 0.1);
//
//    std::cout << nn->predict({0}) << std::endl;
//    std::cout << nn->predict({1}) << std::endl;
//    return 0;
//}
