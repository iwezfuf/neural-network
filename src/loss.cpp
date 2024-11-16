#include "loss.h"
#include <cmath>

float cross_entropy(std::vector<float> predictions, int target) {
    return -log(predictions[target]);
}
