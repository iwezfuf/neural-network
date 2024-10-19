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

    void operator-=(matrix other);

    void operator+=(matrix other);

    matrix operator*(double scalar);

    void randomize();

    void zero();

    std::vector<double> calc_potentials(const std::vector<double> &vec);
};

void vec_apply(std::vector<double> &vec, const std::function<double(double)>& func);

std::vector<double> vec_elementwise_mul(std::vector<double>& vec1, std::vector<double>& vec2);

double sample_normal_dist(double mean, double stddev);

matrix outer_product(const std::vector<double> &vec1, const std::vector<double> &vec2);

std::vector<double> compute_de_dy(const std::vector<double> &prev_de_dy, const std::vector<double> &potential_der, const matrix &weights);

#endif //NEURAL_NETWORK_MATRIX_H
