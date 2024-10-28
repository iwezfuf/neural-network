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
    auto *nn = new neural_network({2, 4, 2}, {Activation::RELU, Activation::SOFTMAX});
    matrix inputs(4, 2);
    inputs.data = {0, 0, 1, 0, 0, 1, 1, 1};
    std::vector<int> labels = {0, 0, 0, 1};
    // print initial weights
//    std::cout << "Initial weights" << std::endl;
//    nn->visualize();
    nn->train(inputs, labels, 1000, 0.05, false);
    // print final weights
//    std::cout << "Final weights" << std::endl;
//    nn->visualize();

    std::vector<int> predicted = {nn->predict((std::vector<double>) {0, 0}), nn->predict(matrix_row_view({1, 0})), nn->predict(matrix_row_view((std::vector<double>) {0, 1})), nn->predict(matrix_row_view({1, 1}))};
    if (predicted[0] == 0 && predicted[1] == 0 && predicted[2] == 0 && predicted[3] == 1) {
        std::cout << "AND test passed" << std::endl;
    } else {
        std::cout << "AND test failed" << std::endl;
    }
    for (size_t i = 0; i < labels.size(); i++) {
//        std::cout << "Expected: " << labels[i] << " Logit for 0: " << nn->logits(inputs.data[i])[0] << " Logit for 1: " << nn->logits(inputs.data[i])[1] << " Predicted: " << predicted[i] << std::endl;
    }
}

void test_or() {
    auto *nn = new neural_network({2, 4, 2}, {Activation::RELU, Activation::SOFTMAX});
    matrix inputs(4, 2);
    inputs.data = {0, 0, 1, 0, 0, 1, 1, 1};
    std::vector<int> labels = {0, 1, 1, 1};
    nn->train(inputs, labels, 1000, 0.05, false);

    std::vector<int> predicted = {nn->predict((std::vector<double>) {0, 0}), nn->predict(matrix_row_view({1, 0})), nn->predict(matrix_row_view((std::vector<double>) {0, 1})), nn->predict(matrix_row_view({1, 1}))};
    if (predicted[0] == 0 && predicted[1] == 1 && predicted[2] == 1 && predicted[3] == 1) {
        std::cout << "OR test passed" << std::endl;
    } else {
        std::cout << "OR test failed" << std::endl;
    }
    for (size_t i = 0; i < labels.size(); i++) {
//        std::cout << "Expected: " << labels[i] << " Logit for 0: " << nn->logits(inputs.data[i])[0] << " Logit for 1: " << nn->logits(inputs.data[i])[1] << " Predicted: " << predicted[i] << std::endl;
    }
}

void test_xor() {
    auto *nn = new neural_network({2, 8, 8, 2}, {Activation::RELU, Activation::RELU, Activation::SOFTMAX});

    matrix inputs(4, 2);
    inputs.data = {0, 0, 1, 0, 0, 1, 1, 1};
    std::vector<int> labels = {0, 1, 1, 0};

//    matrix inputs(1, 2);
//    inputs.data = {{1, 0}};
//    std::vector<int> labels = {1};

    nn->train(inputs, labels, 1000, 0.05, false);

    std::vector<int> predicted = {nn->predict((std::vector<double>) {0, 0}), nn->predict(matrix_row_view({1, 0})), nn->predict(matrix_row_view((std::vector<double>) {0, 1})), nn->predict(matrix_row_view({1, 1}))};
    if (predicted[0] == 0 && predicted[1] == 1 && predicted[2] == 1 && predicted[3] == 0) {
        std::cout << "XOR test passed" << std::endl;
    } else {
        std::cout << "XOR test failed" << std::endl;
    }
    for (size_t i = 0; i < labels.size(); i++) {
//        std::cout << "Expected: " << labels[i] << " Logit for 0: " << nn->logits(inputs.data[i])[0] << " Logit for 1: " << nn->logits(inputs.data[i])[1] << " Predicted: " << predicted[i] << std::endl;
    }
}

void test_larger() {
    auto *nn = new neural_network({2, 2}, {Activation::SOFTMAX});
    matrix inputs(7, 2);
    inputs.data = {0, 1, 1, 0, 2, 3, 3, 2, 1, 2, 2, 1, 100, 10};
    std::vector<int> labels = {0, 1, 0, 1, 0, 1, 1};
    nn->train(inputs, labels, 500, 0.05, false);

    //  TODO overload predict to avoid this
    std::vector<int> predicted = {nn->predict(matrix_row_view( (std::vector<double>) {0, 1})),
                                  nn->predict(matrix_row_view({1, 0})),
                                  nn->predict(matrix_row_view({2, 3})),
                                  nn->predict(matrix_row_view({3, 2})),
                                  nn->predict(matrix_row_view({1, 2})),
                                  nn->predict(matrix_row_view({2, 1})),
                                  nn->predict(matrix_row_view({100, 10}))};
    if (predicted[0] == 0 && predicted[1] == 1 && predicted[2] == 0 && predicted[3] == 1 && predicted[4] == 0 && predicted[5] == 1 && predicted[6] == 1) {
        std::cout << "Larger test passed" << std::endl;
    } else {
        std::cout << "Larger test failed, outputs: " << std::endl;
    }
    for (size_t i = 0; i < labels.size(); i++) {
//        std::cout << "Expected: " << labels[i] << " Logit for 0: " << nn->logits(inputs.data[i])[0] << " Logit for 1: " << nn->logits(inputs.data[i])[1] << " Predicted: " << predicted[i] << std::endl;
    }
}

void dataset() {
    std::string labels_file = "../data/fashion_mnist_train_labels.csv";
    std::string vectors_file = "../data/fashion_mnist_train_vectors.csv";
    if (true) {
        labels_file = "../data/mnist_digits/train_labels.csv";
        vectors_file = "../data/mnist_digits/train_images.csv";
    }

    std::vector<int> labels;
    std::vector<double> vectors;

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
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            vectors.push_back(std::stod(value));
        }
    }
    vectors_stream.close();

    std::cout << "Loaded " << labels.size() << " labels and " << vectors.size() / 784 << " vectors." << std::endl;

    auto *nn = new neural_network({784, 200, 80, 10}, {Activation::RELU, Activation::RELU, Activation::SOFTMAX});
    matrix inputs(vectors, static_cast<int>(vectors.size() / 784), 784);
    inputs.normalize_data();

    std::cout << "Correct before train: " << nn->correct(inputs, labels) << std::endl;

    nn->train(inputs, labels, 60000/32, 0.001, false);

    std::cout << "Correct after train: " << nn->correct(inputs, labels) << std::endl;
}

int main() {
//    test_and();
//    test_or();
//    test_larger();
//    test_xor();

    dataset();
    return 0;
}
