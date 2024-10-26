#include <cmath>

#include "optimizers.h"
#include "matrix.h"

void adam_optimizer::update_weights(matrix& weights, matrix& weights_delta, double learning_rate) {
    t++;

    // update v
    for (int i = 0; i < v.size(); i++) {
        v.data[i] = v.data[i] * beta2 + weights_delta.data[i] * weights_delta.data[i] * (1 - beta2);
    }

    // Update weights
    for (int i = 0; i < weights.size(); i++) {
        double weights_delta_corrected = weights_delta.data[i] / (1 - pow(beta1, t));
        double v_corrected = v.data[i] / (1 - pow(beta2, t));
        weights.data[i] -= learning_rate * weights_delta_corrected / (sqrt(v_corrected) + epsilon);
    }

    // update weights_delta for next iteration
    weights_delta.multiply_by_scalar(beta1);
}

void adam_optimizer::add_current_example_weight_gradient(matrix &weights_delta, const matrix &current_weights_delta) {
    weights_delta += current_weights_delta * (1 - beta1);
}
