#include <cmath>

#include "optimizers.h"
#include "matrix.h"

#define no_optimizer true

void adam_optimizer::update_weights(matrix& weights, matrix& weights_delta, float learning_rate) {
    // TODO
    if (no_optimizer) {
        for (int i = 0; i < weights.size(); i++) {
            weights.data[i] -= learning_rate * weights_delta.data[i];
        }
        weights_delta.zero();
        return;
    }
    t++;

    // update v
    for (int i = 0; i < v.size(); i++) {
        v.data[i] = v.data[i] * beta2 + weights_delta.data[i] * weights_delta.data[i] * (1 - beta2);
    }

    // Update weights
    for (int i = 0; i < weights.size(); i++) {
        float weights_delta_corrected = weights_delta.data[i] / (1 - pow(beta1, t));
        float v_corrected = v.data[i] / (1 - pow(beta2, t));
        weights.data[i] -= learning_rate * weights_delta_corrected / (sqrt(v_corrected) + epsilon);

//        weights.data[i] -= learning_rate * weights_delta.data[i];
    }

    // update weights_delta for next iteration
    weights_delta.multiply_by_scalar(1 - beta1);
}

void adam_optimizer::add_current_example_weight_gradient(matrix &weights_delta, const matrix &current_weights_delta) {
    if (no_optimizer) {
        for (int i = 0; i < weights_delta.size(); i++) {
            weights_delta.data[i] += current_weights_delta.data[i];
        }
        return;
    }
    add_scaled_vector(weights_delta.data, current_weights_delta.data, beta1);
}
