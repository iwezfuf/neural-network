#ifndef NEURAL_NETWORK_MATRIX_H
#define NEURAL_NETWORK_MATRIX_H

#include <vector>
#include <functional>

struct matrix {
    int rows;
    int cols;
    std::vector<std::vector<double>> data;
public:
    matrix(int rows, int cols);

    matrix(std::vector<std::vector<double>> data);

    std::vector<double> operator*(std::vector<double> vec);

    matrix operator*(matrix other);

    void operator-=(matrix other);

    matrix operator*(double scalar);

    void substract(matrix other);

    void randomize();

    matrix transposed();

    matrix without_last_col();
};

void vec_mul_scalar(std::vector<double> vec, double scalar);

void vec_apply(std::vector<double> &vec, std::function<double(double)> func);

std::vector<double> vec_elementwise_mul(std::vector<double>& vec1, std::vector<double>& vec2);

matrix matrix_mul(matrix& m1, matrix& m2);

double sample_normal_dist(double mean, double stddev);

matrix outer_product(std::vector<double> &vec1, std::vector<double> &vec2);

#endif //NEURAL_NETWORK_MATRIX_H
