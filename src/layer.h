#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include <functional>
#include "matrix.h"

struct layer {
    int size;
    std::function<float(float)> activation;
    matrix weights;
};


#endif //NEURAL_NETWORK_LAYER_H
