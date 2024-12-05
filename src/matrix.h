#ifndef NEURAL_NETWORK_MATRIX_H
#define NEURAL_NETWORK_MATRIX_H

#include <vector>
#include <functional>
#include <utility>
#include <vector>
#include <cassert>
#include <random>
#include <iostream>

inline float sample_normal_dist(float mean, float stddev, std::mt19937 &gen) {
    std::normal_distribution<> dist(mean, stddev);
    return static_cast<float>(dist(gen));
}

class matrix_row_view {
private:
    const float* data;
public:
    size_t size;

    matrix_row_view(const float* data, size_t size) : data(data), size(size) {}

    matrix_row_view(const std::vector<float>& vec) : data(vec.data()), size(vec.size()) {}

    float& operator[](size_t index);

    inline const float& operator[](size_t index) const {
        return data[index];
    }

    const float* begin() { return data; }
    const float* end() { return data + size; }
    const float* begin() const { return data; }
    const float* end() const { return data + size; }

    size_t length() const { return size; }
};


struct matrix {
    inline matrix(std::vector<float> data, int rows, int cols) {
        this->rows = rows;
        this->cols = cols;
        this->data = std::move(data);
    }

    int rows;
    int cols;
    std::vector<float> data;
public:
    inline matrix(int rows, int cols) {
        this->rows = rows;
        this->cols = cols;
        data.resize(rows * cols, 0);
    }

    inline void operator-=(const matrix &other) {
        for (int i = 0; i < size(); i++) {
            data[i] -= other.data[i];
        }
    }

    inline void operator+=(const matrix &other) {
        for (int i = 0; i < size(); i++) {
            data[i] += other.data[i];
        }
    }

    inline void randomize() {
        unsigned int seed = std::random_device("/dev/random")();
        std::cout << "Seed used in randomize: " << seed << std::endl;
        auto gen = std::mt19937(seed);
        // use He initialization
        for (int i = 0; i < size(); i++) {
            // leave bias at 0
            if ((i + 1) % cols != 0) {
                data[i] = sample_normal_dist(0, sqrt(2.0 / cols), gen);
            }
        }
    }

    inline void zero() {
        std::fill(data.begin(), data.end(), 0);
    }

    inline std::vector<float> calc_potentials(const matrix_row_view &vec) {
        std::vector<float> result;
        result.reserve(rows);

        for (int i = 0; i < rows; i++) {
            float sum = 0;
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

    inline int index(int row, int col) const {
        return row * cols + col;
    }

    inline matrix_row_view get_row(int row)  const {
        return matrix_row_view(&data[row * cols], cols);
    }

    inline std::vector<std::pair<float, float>> calc_mean_stddev() {
        std::vector<std::pair<float, float>> result;
        result.reserve(cols - 1);
        for (int i = 0; i < cols - 1; i++) {
            float sum = 0;
            for (int j = 0; j < rows; j++) {
                sum += data[index(j, i)];
            }
            float mean = sum / rows;
            float stddev = 0;
            for (int j = 0; j < rows; j++) {
                stddev += (data[index(j, i)] - mean) * (data[index(j, i)] - mean);
            }
            stddev = sqrt(stddev / rows);
            result.push_back(std::make_pair(mean, stddev));
        }
        return result;
    }

    inline void normalize_data(std::vector<std::pair<float, float>> mean_stddev) {
        for (int i = 0; i < cols - 1; i++) {
            for (int j = 0; j < rows; j++) {
                data[index(j, i)] = (data[index(j, i)] - mean_stddev[i].first) / mean_stddev[i].second;
            }
        }
    }

    inline void multiply_by_scalar(float scalar) {
        for (int i = 0; i < size(); i++) {
            data[i] *= scalar;
        }
    }

    int size() const { return rows * cols; }
};

inline void vec_apply(std::vector<float> &vec, const std::function<float(float)>& func) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = func(vec[i]);
    }
}

inline std::vector<float> vec_elementwise_mul(std::vector<float>& vec1, std::vector<float>& vec2) {
    assert(vec1.size() == vec2.size());
    std::vector<float> result;
    result.reserve(vec1.size());
    for (size_t i = 0; i < vec1.size(); i++) {
        result.push_back(vec1[i] * vec2[i]);
    }
    return result;
}

inline matrix outer_product(const matrix_row_view &vec1, const matrix_row_view &vec2) {
    matrix result(static_cast<int>(vec1.size), static_cast<int>(vec2.size + 1));
    for (size_t i = 0; i < vec1.size; i++) {
        for (size_t j = 0; j < vec2.size; j++) {
            result.data[result.index(static_cast<int>(i), static_cast<int>(j))] = vec1[i] * vec2[j];
        }
    }
    for (size_t i = 0; i < vec1.size; i++) {
        result.data[result.index(static_cast<int>(i), static_cast<int>(vec2.size))] = vec1[i];
    }
    return result;
}

inline std::vector<float> compute_de_dy(const std::vector<float> &prev_de_dy, const std::vector<float> &potential_der, const matrix &weights) {
    std::vector<float> res(weights.cols - 1, 0);

    for (size_t i = 0; i < prev_de_dy.size(); i++) {
        float factor = prev_de_dy[i] * potential_der[i];
        // skip bias
        for (int j = 0; j < weights.cols - 1; j++) {
            res[j] += factor * weights.data[weights.index(static_cast<int>(i), j)];
        }
    }
    return res;
}

inline void add_scaled_vector(std::vector<float> &vec, const std::vector<float> &vec_to_add, float scale)  {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] += vec_to_add[i] * scale;
    }
}

#endif //NEURAL_NETWORK_MATRIX_H
