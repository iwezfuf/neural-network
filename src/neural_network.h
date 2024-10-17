#ifndef NEURAL_NETWORK_NEURAL_NETWORK_H
#define NEURAL_NETWORK_NEURAL_NETWORK_H

#include <vector>
#include <functional>

#include "matrix.h"
#include "layer.h"

struct neural_network {
    std::vector<layer> layers;
    std::function<double(std::vector<double>, std::vector<double>)> error_function;
public:
    neural_network(std::vector<int> sizes,
                   std::vector<std::function<void(std::vector<double>&)>> activations,
                   std::vector<std::function<void(std::vector<double>&)>> activation_derivatives);

    std::vector<double> logits(const std::vector<double> &input);

    int predict(const std::vector<double>& input);

    void visualize();

    void forward(const std::vector<double> &input);

    std::vector<double>& get_outputs();

    void train(const matrix &inputs, const std::vector<int> &labels, int epochs, double learning_rate, bool debug_mode);

//    void backward(const std::vector<double> &input, const std::vector<double> &label);

    void backward(std::vector<double> input, std::vector<double> label);
};


#endif //NEURAL_NETWORK_NEURAL_NETWORK_H
