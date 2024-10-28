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
    neural_network(std::vector<int> sizes, const std::vector<Activation> &activations);

    std::vector<double> logits(const matrix_row_view &input);

    int predict(const matrix_row_view& input);

    void visualize();

    void forward(const matrix_row_view &input);

    std::vector<double>& get_outputs();

    void train(const matrix &inputs, const std::vector<int> &labels, int epochs, double learning_rate, bool debug_mode);

    void backward(const matrix_row_view &input, const std::vector<double> &label);

    int accuracy(const matrix &inputs, const std::vector<int> &labels);
};


#endif //NEURAL_NETWORK_NEURAL_NETWORK_H
