#include <vector>
#include <functional>

#include "matrix.h"


matrix::matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    data.resize(rows);
    for (int i = 0; i < rows; i++) {
        data[i].resize(cols, 0);
    }
}

matrix::matrix(std::vector<std::vector<float>> data) {
    this->data = data;
    this->rows = data.size();
    this->cols = data[0].size();
}

std::vector<float> matrix::operator*(std::vector<float> vec) {
    std::vector<float> result;
    for (int i = 0; i < rows; i++) {
        float sum = 0;
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
            float sum = 0;
            for (int k = 0; k < cols; k++) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    return result;
}

void matrix::randomize() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = static_cast<float>(rand()) / RAND_MAX * 2 - 1;
        }
    }
}


void vec_mul_scalar(std::vector<float> vec, float scalar) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] *= scalar;
    }
}

void vec_apply(std::vector<float> vec, std::function<float(float)> func) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = func(vec[i]);
    }
}

std::vector<float> vec_elementwise_mul(std::vector<float>& vec1, std::vector<float>& vec2) {
    std::vector<float> result;
    for (size_t i = 0; i < vec1.size(); i++) {
        result.push_back(vec1[i] * vec2[i]);
    }
    return result;
}

