#ifndef NEURAL_NETWORK_NEURAL_NETWORK_H
#define NEURAL_NETWORK_NEURAL_NETWORK_H

#include <vector>
#include <functional>

#include "matrix.h"
#include "layer.h"

struct neural_network {
    std::vector<layer> layers;
    std::function<float(std::vector<float>, std::vector<float>)> error_function;
public:
    neural_network(std::vector<int> sizes, const std::vector<Activation> &activations);

    int predict(const matrix_row_view& input);

    void visualize();

    void forward(const matrix_row_view &input);

    std::vector<float>& get_outputs();

    void train(const matrix &inputs, const std::vector<int> &labels, int epochs, float learning_rate, int batch_size);

    void backward(const matrix_row_view &input, const std::vector<float> &label);

    float accuracy(const matrix &inputs, const std::vector<int> &labels);

    void predict_to_file(const matrix &inputs, const std::string& filename);
};


#endif //NEURAL_NETWORK_NEURAL_NETWORK_H
