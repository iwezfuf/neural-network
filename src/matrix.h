#ifndef NEURAL_NETWORK_MATRIX_H
#define NEURAL_NETWORK_MATRIX_H

#include <vector>
#include <functional>

class matrix_row_view {
private:
    const double* data;
public:
    size_t size;

    matrix_row_view(const double* data, size_t size) : data(data), size(size) {}

    matrix_row_view(const std::vector<double>& vec) : data(vec.data()), size(vec.size()) {}

    double& operator[](size_t index);

    const double& operator[](size_t index) const;

    const double* begin() { return data; }
    const double* end() { return data + size; }
    const double* begin() const { return data; }
    const double* end() const { return data + size; }

    size_t length() const { return size; }
};


struct matrix {
    matrix(std::vector<double> data, int rows, int cols);

    int rows;
    int cols;
    std::vector<double> data;
public:
    matrix(int rows, int cols);

    void operator-=(matrix other);

    void operator+=(matrix other);

    matrix operator*(double scalar);

    void randomize();

    void zero();

    std::vector<double> calc_potentials(const matrix_row_view &vec);

    inline int index(int row, int col) const;

    matrix_row_view get_row(int row) const;

    void normalize_data();
};

void vec_apply(std::vector<double> &vec, const std::function<double(double)>& func);

std::vector<double> vec_elementwise_mul(std::vector<double>& vec1, std::vector<double>& vec2);

double sample_normal_dist(double mean, double stddev);

matrix outer_product(const matrix_row_view &vec1, const matrix_row_view &vec2);

std::vector<double> compute_de_dy(const std::vector<double> &prev_de_dy, const std::vector<double> &potential_der, const matrix &weights);

#endif //NEURAL_NETWORK_MATRIX_H
