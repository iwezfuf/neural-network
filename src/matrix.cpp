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
