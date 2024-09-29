#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include <functional>
#include <memory>
#include <vector>

#include "matrix.h"

struct layer {
    int size;
    int size_incoming;
    std::function<void(std::vector<double>&)> activation;
    std::function<void(std::vector<double>&)> activation_derivative;
    std::unique_ptr<matrix> weights;
    std::unique_ptr<std::vector<double>> values;
    std::unique_ptr<std::vector<double>> potential_der;

public:
    layer(int size, int size_incoming,
          std::function<void(std::vector<double>&)> activation,
          std::function<void(std::vector<double>&)> activation_derivative);

    std::vector<double> forward(std::vector<double> input);
};


#endif //NEURAL_NETWORK_LAYER_H
