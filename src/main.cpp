#include <iostream>
#include <vector>
#include "neural_network.h"
#include "matrix.h"
#include "activation.h"

//int main() {
//    auto *nn = new neural_network({2, 4, 4, 2}, {relu, relu, softmax}, {relu_derivative, relu_derivative, softmax_derivative});
//
//    matrix inputs(4, 2);
//    inputs.data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
//
//    std::vector<int> labels = {0, 1, 1, 0};
//
//    nn->train(inputs, labels, 1000, 0.05);
//
//    std::cout << nn->predict({0, 0}) << std::endl;
//    std::cout << nn->predict({0, 1}) << std::endl;
//    std::cout << nn->predict({1, 0}) << std::endl;
//    std::cout << nn->predict({1, 1}) << std::endl;
//    return 0;
//}

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

void test_larger() {
    auto *nn = new neural_network({2, 2}, {softmax}, {softmax_derivative});
    matrix inputs(4, 2);
    inputs.data = {{0, 1}, {1, 0}, {2, 3}, {3, 2}};
    std::vector<int> labels = {0, 1, 0, 1};
    nn->train(inputs, labels, 100, 0.1);

    std::vector<int> predicted = {nn->predict({0, 1}), nn->predict({1, 0}), nn->predict({2, 3}), nn->predict({3, 2})};
    if (predicted[0] == 0 && predicted[1] == 1 && predicted[2] == 0 && predicted[3] == 1) {
        std::cout << "Larger test passed" << std::endl;
    } else {
        std::cout << "Larger test failed, outputs: " << std::endl;
        for (auto& val : predicted) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

}

int main() {
    test_and();
    test_or();
    test_larger();

    auto *nn = new neural_network({2, 1, 2}, {relu, softmax}, {relu_derivative, softmax_derivative});

    matrix inputs(4, 2);
    inputs.data = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

    std::vector<int> labels = {0, 1, 1, 0};

    nn->train(inputs, labels, 500, 0.05);

    std::vector<int> predicted = {nn->predict({0, 0}), nn->predict({1, 0}), nn->predict({0, 1}), nn->predict({1, 1})};
    for (auto& val : predicted) {
        std::cout << val << " ";
    }
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
