#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include <functional>
#include <memory>
#include <vector>

#include "matrix.h"
#include "activation.h"
#include "optimizers.h"

struct layer {
    int size;
    Activation activation;
    std::unique_ptr<matrix> weights;
    std::unique_ptr<matrix> weights_delta;
    std::unique_ptr<std::vector<float>> values;
    std::unique_ptr<std::vector<float>> potential_der;
    std::unique_ptr<adam_optimizer> optimizer;

public:
    layer(int size, int size_incoming, Activation activation);

    void forward(const matrix_row_view &input);

    void update_weights(float learning_rate) const;

    void zero_weights_delta() const;
};


#endif //NEURAL_NETWORK_LAYER_H
