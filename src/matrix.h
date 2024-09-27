#ifndef NEURAL_NETWORK_MATRIX_H
#define NEURAL_NETWORK_MATRIX_H

#include <vector>
#include <functional>

struct matrix {
    int rows;
    int cols;
    std::vector<std::vector<float>> data;
public:
    matrix(int rows, int cols);

    std::vector<float> operator*(std::vector<float> vec);

    void operator-=(matrix other);
};

void vec_mul_scalar(std::vector<float> vec, float scalar);

void vec_apply(std::vector<float> vec, std::function<float(float)> func);

#endif //NEURAL_NETWORK_MATRIX_H
