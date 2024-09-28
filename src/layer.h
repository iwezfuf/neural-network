#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include <functional>
#include <memory>
#include <vector>

#include "matrix.h"

struct layer {
    int size;
    int size_incoming;
    std::function<void(std::vector<float>&)> activation;
    std::function<void(std::vector<float>&)> activation_derivative;
    std::unique_ptr<matrix> weights;
    std::unique_ptr<std::vector<float>> values;
    std::unique_ptr<std::vector<float>> potential;

public:
    layer(int size, int size_incoming,
          std::function<void(std::vector<float>&)> activation,
          std::function<void(std::vector<float>&)> activation_derivative);

    std::vector<float> forward(std::vector<float> input);
};


#endif //NEURAL_NETWORK_LAYER_H
