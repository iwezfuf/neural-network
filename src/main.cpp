#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "neural_network.h"
#include "matrix.h"
#include "activation.h"

void load_dataset(const std::string& labels_file, const std::string& vectors_file, std::vector<int> &labels, std::vector<float> &vectors) {
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
            vectors.push_back(std::stof(value));
        }
    }
    vectors_stream.close();
}

void train_mnist() {
    std::string labels_file = "data/fashion_mnist_train_labels.csv";
    std::string vectors_file = "data/fashion_mnist_train_vectors.csv";
    std::string test_labels_file = "data/fashion_mnist_test_labels.csv";
    std::string test_vectors_file = "data/fashion_mnist_test_vectors.csv";

//    labels_file = "data/mnist_digits/train_labels.csv";
//    vectors_file = "data/mnist_digits/train_images.csv";
//    test_labels_file = "data/mnist_digits/test_labels.csv";
//    test_vectors_file = "data/mnist_digits/test_images.csv";

    std::vector<int> labels;
    std::vector<float> vectors;
    load_dataset(labels_file, vectors_file, labels, vectors);

    std::vector<int> test_labels;
    std::vector<float> test_vectors;
    load_dataset(test_labels_file, test_vectors_file, test_labels, test_vectors);

    std::cout << "Loaded " << labels.size() << " labels and " << vectors.size() / 784 << " vectors." << std::endl;
    std::cout << "Loaded " << test_labels.size() << " test labels and " << test_vectors.size() / 784 << " test vectors." << std::endl;

    auto *nn = new neural_network({784, 256, 128, 10}, {Activation::RELU, Activation::RELU, Activation::SOFTMAX});

    matrix inputs(vectors, static_cast<int>(vectors.size() / 784), 784);
    std::vector<std::pair<float, float>> mean_stddev = inputs.calc_mean_stddev();
    inputs.normalize_data(mean_stddev);

    matrix test_inputs(test_vectors, static_cast<int>(test_vectors.size() / 784), 784);
    test_inputs.normalize_data(mean_stddev);

    nn->train(inputs, labels, 6, 0.001, 32);

    nn->predict_to_file(test_inputs, "predictions.csv");
    // std::cout << "Accuracy train: " << nn->accuracy(inputs, labels) << "%" << std::endl;
    // std::cout << "Accuracy test:  " << nn->accuracy(test_inputs, test_labels) << "%" << std::endl;
}

int main() {
    train_mnist();
    return 0;
}
