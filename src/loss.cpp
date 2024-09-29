#include "loss.h"
#include <cmath>

double cross_entropy(std::vector<double> predictions, int target) {
    return -log(predictions[target]);
}
