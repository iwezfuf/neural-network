#include <vector>
#include <functional>
#include <cassert>
#include <cmath>

#include "matrix.h"

matrix::matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    data.resize(rows);
    for (int i = 0; i < rows; i++) {
        data[i].resize(cols, 0);
    }
}

matrix::matrix(std::vector<std::vector<double>> data) {
    this->data = data;
    this->rows = data.size();
    this->cols = data[0].size();
}

std::vector<double> matrix::operator*(std::vector<double> vec) {
    std::vector<double> result;
    for (int i = 0; i < rows; i++) {
        double sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += data[i][j] * vec[j];
        }
        result.push_back(sum);
    }
    return result;
}

void matrix::operator-=(matrix other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] -= other.data[i][j];
        }
    }
}

matrix matrix::operator*(matrix other) {
    matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            double sum = 0;
            for (int k = 0; k < cols; k++) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    return result;
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

void matrix::substract(matrix other) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] -= other.data[i][j];
        }
    }
}

void matrix::randomize() {
    // use glorot initialization
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = sample_normal_dist(0, 2.0 / (rows + cols));
        }
    }
}

matrix matrix::transposed() {
    matrix result(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

matrix matrix::without_last_col() {
    matrix result(rows, cols - 1);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols - 1; j++) {
            result.data[i][j] = data[i][j];
        }
    }
    return result;
}

void matrix::zero() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = 0;
        }
    }
}


void vec_mul_scalar(std::vector<double> vec, double scalar) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] *= scalar;
    }
}

void vec_apply(std::vector<double> &vec, std::function<double(double)> func) {
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
    double u = (double) rand() / RAND_MAX;
    double v = (double) rand() / RAND_MAX;
    double z = sqrt(-2 * log(u)) * cos(2 * M_PI * v);
    return mean + z * stddev;
}

matrix outer_product(std::vector<double> &vec1, std::vector<double> &vec2) {
    matrix result(vec1.size(), vec2.size());
    for (size_t i = 0; i < vec1.size(); i++) {
        for (size_t j = 0; j < vec2.size(); j++) {
            result.data[i][j] = vec1[i] * vec2[j];
        }
    }
    return result;
}
