#include "loss.h"
#include <cmath>

double cross_entropy(std::vector<float> predictions, int target) {
    return -log(predictions[target]);
}
