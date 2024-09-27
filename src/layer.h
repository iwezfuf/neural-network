#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include <functional>
#include <memory>

#include "matrix.h"

struct layer {
    int size;
    int size_incoming;
    std::function<float(float)> activation;
    std::unique_ptr<matrix> weights;

public:
    layer(int size, int size_incoming, std::function<float(float)> activation);

    std::vector<float> forward(std::vector<float> input);
};


#endif //NEURAL_NETWORK_LAYER_H
