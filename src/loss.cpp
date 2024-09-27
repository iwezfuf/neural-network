#include "loss.h"
#include <cmath>

double cross_entropy(std::vector<float> predictions, int target) {
    float loss = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        loss += static_cast<size_t>(target) == i ? -log(predictions[i]) : -log(1 - predictions[i]);
    }
    return loss;
}
