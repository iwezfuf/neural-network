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

    matrix(std::vector<std::vector<float>> data);

    std::vector<float> operator*(std::vector<float> vec);

    void operator-=(matrix other);

    matrix operator*(matrix other);

    void randomize();
};

void vec_mul_scalar(std::vector<float> vec, float scalar);

void vec_apply(std::vector<float> vec, std::function<float(float)> func);

std::vector<float> vec_elementwise_mul(std::vector<float>& vec1, std::vector<float>& vec2);

matrix matrix_mul(matrix& m1, matrix& m2);


#endif //NEURAL_NETWORK_MATRIX_H
