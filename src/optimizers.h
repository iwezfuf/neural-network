#ifndef MAINPROJECT_OPTIMIZERS_H
#define MAINPROJECT_OPTIMIZERS_H

struct adam_optimizer {
    int t = 0;
    matrix v;

    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double epsilon = 1e-8;
public:
    adam_optimizer(int rows, int cols) : v(rows, cols) {
        v.zero();
    }

    void update_weights(matrix& weights, matrix& weights_delta, double learning_rate);
    void add_current_example_weight_gradient(matrix &weights_delta, const matrix &current_weights_delta);
};


#endif //MAINPROJECT_OPTIMIZERS_H
