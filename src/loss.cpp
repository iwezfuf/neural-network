#include "loss.h"
#include <cmath>

double cross_entropy(std::vector<float> predictions, int target) {
    float loss;
    for (int i = 0; i < predictions.size(); i++) {
        loss += target == i ? -log(predictions[i]) : -log(1 - predictions[i]);
    }
    return loss;
}
