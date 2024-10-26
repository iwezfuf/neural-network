#include <utility>
#include <vector>
#include <cassert>
#include <random>

#include "matrix.h"

const double& matrix_row_view::operator[](size_t index) const {
    return data[index];
}

matrix::matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    data.resize(rows * cols, 0);
}

matrix::matrix(std::vector<double> data, int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = std::move(data);
}

void matrix::operator-=(matrix other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[index(i, j)] -= other.data[index(i, j)];
        }
    }
}

void matrix::operator+=(matrix other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[index(i, j)] += other.data[index(i, j)];
        }
    }
}

void matrix::randomize() {
    // use He initialization, set bias to 0
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols - 1; j++) {
            data[index(i, j)] = sample_normal_dist(0, sqrt(2.0 / cols));
        }
    }
}

void matrix::zero() {
    std::fill(data.begin(), data.end(), 0);
}

std::vector<double> matrix::calc_potentials(const matrix_row_view &vec) {
    std::vector<double> result;
    result.reserve(rows);

    for (int i = 0; i < rows; i++) {
        double sum = 0;
        for (int j = 0; j < cols - 1; j++) {
            sum += data[index(i, j)] * vec[j];
        }
        result.push_back(sum);
    }
    // add biases
    for (int i = 0; i < rows; i++) {
        result[i] += data[index(i, cols - 1)];
    }
    return result;
}

void vec_apply(std::vector<double> &vec, const std::function<double(double)>& func) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = func(vec[i]);
    }
}

std::vector<double> vec_elementwise_mul(std::vector<double>& vec1, std::vector<double>& vec2) {
    assert(vec1.size() == vec2.size());
    std::vector<double> result;
    result.reserve(vec1.size());
    for (size_t i = 0; i < vec1.size(); i++) {
        result.push_back(vec1[i] * vec2[i]);
    }
    return result;
}

double sample_normal_dist(double mean, double stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(mean, stddev);
    return dist(gen);
}

matrix outer_product(const matrix_row_view &vec1, const matrix_row_view &vec2) {
    matrix result(static_cast<int>(vec1.size), static_cast<int>(vec2.size + 1));
    for (size_t i = 0; i < vec1.size; i++) {
        for (size_t j = 0; j < vec2.size; j++) {
            result.data[result.index(i, j)] = vec1[i] * vec2[j];
        }
    }
    for (size_t i = 0; i < vec1.size; i++) {
        result.data[result.index(i, vec2.size)] = vec1[i];
    }
    return result;
}

std::vector<double> compute_de_dy(const std::vector<double> &prev_de_dy, const std::vector<double> &potential_der, const matrix &weights) {
    std::vector<double> res(weights.cols - 1, 0);

    for (size_t i = 0; i < prev_de_dy.size(); i++) {
        double factor = prev_de_dy[i] * potential_der[i];
        // skip bias
        for (int j = 0; j < weights.cols - 1; j++) {
            res[j] += factor * weights.data[weights.index(i, j)];
        }
    }
    return res;
}

matrix_row_view matrix::get_row(int row) const {
    return matrix_row_view(&data[row * cols], cols);
}

inline int matrix::index(int row, int col) const {
    return row * cols + col;
}

void matrix::normalize_data() {
    // substract mean and divide by standard deviation
    for (int i = 0; i < cols - 1; i++) {
        double sum = 0;
        for (int j = 0; j < rows; j++) {
            sum += data[index(j, i)];
        }
        double mean = sum / rows;
        double stddev = 0;
        for (int j = 0; j < rows; j++) {
            stddev += (data[index(j, i)] - mean) * (data[index(j, i)] - mean);
        }
        stddev = sqrt(stddev / rows);
        for (int j = 0; j < rows; j++) {
            data[index(j, i)] = (data[index(j, i)] - mean) / stddev;
        }
    }
}

void matrix::multiply_by_scalar(double scalar) {
    for (int i = 0; i < size(); i++) {
        data[i] *= scalar;
    }
}

void add_scaled_vector(std::vector<double> &vec, const std::vector<double> &vec_to_add, double scale) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] += vec_to_add[i] * scale;
    }
}