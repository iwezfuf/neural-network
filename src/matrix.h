#ifndef NEURAL_NETWORK_MATRIX_H
#define NEURAL_NETWORK_MATRIX_H

#include <vector>
#include <functional>

class matrix_row_view {
private:
    const float* data;
public:
    size_t size;

    matrix_row_view(const float* data, size_t size) : data(data), size(size) {}

    matrix_row_view(const std::vector<float>& vec) : data(vec.data()), size(vec.size()) {}

    float& operator[](size_t index);

    const float& operator[](size_t index) const;

    const float* begin() { return data; }
    const float* end() { return data + size; }
    const float* begin() const { return data; }
    const float* end() const { return data + size; }

    size_t length() const { return size; }
};


struct matrix {
    matrix(std::vector<float> data, int rows, int cols);

    int rows;
    int cols;
    std::vector<float> data;
public:
    matrix(int rows, int cols);

    void operator-=(matrix other);

    void operator+=(matrix other);

    void randomize();

    void zero();

    std::vector<float> calc_potentials(const matrix_row_view &vec);

    inline int index(int row, int col) const;

    matrix_row_view get_row(int row) const;

    void normalize_data();

    void multiply_by_scalar(float scalar);

    int size() const { return rows * cols; }
};

void vec_apply(std::vector<float> &vec, const std::function<float(float)>& func);

std::vector<float> vec_elementwise_mul(std::vector<float>& vec1, std::vector<float>& vec2);

float sample_normal_dist(float mean, float stddev);

matrix outer_product(const matrix_row_view &vec1, const matrix_row_view &vec2);

std::vector<float> compute_de_dy(const std::vector<float> &prev_de_dy, const std::vector<float> &potential_der, const matrix &weights);

void add_scaled_vector(std::vector<float> &vec, const std::vector<float> &vec_to_add, float scale);

#endif //NEURAL_NETWORK_MATRIX_H
