#include <vector>
#include <cassert>
#include <random>

#include "matrix.h"

matrix::matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    data.resize(rows);
    for (int i = 0; i < rows; i++) {
        data[i].resize(cols, 0);
    }
}

void matrix::operator-=(matrix other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] -= other.data[i][j];
        }
    }
}

void matrix::operator+=(matrix other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] += other.data[i][j];
        }
    }
}

matrix matrix::operator*(double scalar) {
    matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }
    return result;
}

void matrix::randomize() {
    // use He initialization, set bias to 0
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols - 1; j++) {
            data[i][j] = sample_normal_dist(0, sqrt(2.0 / cols));
        }
    }
}

void matrix::zero() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = 0;
        }
    }
}

std::vector<double> matrix::calc_potentials(const std::vector<double> &vec) {
    std::vector<double> result;
    for (int i = 0; i < rows; i++) {
        double sum = 0;
        for (int j = 0; j < cols - 1; j++) {
            sum += data[i][j] * vec[j];
        }
        result.push_back(sum);
    }
    // add biases
    for (int i = 0; i < rows; i++) {
        result[i] += data[i][cols - 1];
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

// vec2 should have an additional 1 at the end
matrix outer_product(const std::vector<double> &vec1, const std::vector<double> &vec2) {
    matrix result(static_cast<int>(vec1.size()), static_cast<int>(vec2.size() + 1));
    for (size_t i = 0; i < vec1.size(); i++) {
        for (size_t j = 0; j < vec2.size(); j++) {
            result.data[i][j] = vec1[i] * vec2[j];
        }
    }
    for (size_t i = 0; i < vec1.size(); i++) {
        result.data[i][vec2.size()] = vec1[i];
    }
    return result;
}

std::vector<double> compute_de_dy(const std::vector<double> &prev_de_dy, const std::vector<double> &potential_der, const matrix &weights) {
    std::vector<double> res(weights.cols - 1, 0);

    for (size_t i = 0; i < prev_de_dy.size(); i++) {
        double factor = prev_de_dy[i] * potential_der[i];
        // skip bias
        for (int j = 0; j < weights.cols - 1; j++) {
            res[j] += factor * weights.data[i][j];
        }
    }
    return res;
}
