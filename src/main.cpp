#include <iostream>
#include <vector>
#include <functional>
#include "neural_network.h"
#include "matrix.h"

int main() {
    auto* a = new matrix(2, 2);

    std::vector<float> vec = {1, 2};

    std::vector<float> result = (*a) * vec;

    std::cout << result[0] << " " << result[1] << std::endl;

    delete a;

    return 0;
}
