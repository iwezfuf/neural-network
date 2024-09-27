#ifndef NEURAL_NETWORK_LOSS_H
#define NEURAL_NETWORK_LOSS_H

#include <vector>
#include "matrix.h"

double cross_entropy(std::vector<float> predictions, int target);
matrix cross_entropy_derivative(std::vector<float> predictions, int target);


#endif //NEURAL_NETWORK_LOSS_H
