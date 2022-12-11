#ifndef _RIDGEREG_H_
#define _RIDGEREG_H_

#include <iostream>
#include <eigen3/Eigen/Dense>
using namespace std;

class LinearModel {
private:
    Eigen::VectorXd weights;
public:
    LinearModel(int num_feature) {
        this->weights = Eigen::MatrixXd::Random(num_feature + 1, 1);
    };
    ~LinearModel() {};
    Eigen::VectorXd get_weights() {return this->weights;};
    void set_weights(Eigen::VectorXd weights) {
        this->weights = weights;
    };
    Eigen::VectorXd forward(Eigen::MatrixXd X);
    double calc_loss(Eigen::VectorXd Y, Eigen::VectorXd Y_hat) {
        Eigen::VectorXd err = Y - Y_hat;
        return (err.transpose() * err)(0, 0) / err.size();
    }
};

Eigen::VectorXd LinearModel::forward(Eigen::MatrixXd X) {
    int X_rows = X.rows();
    int X_cols = X.cols();
    Eigen::MatrixXd X_extended = Eigen::MatrixXd::Constant(X_rows, X_cols + 1, 1);
    X_extended.block(0, 1, X_rows, X_cols) = X;
    return X_extended * this->weights;
}

#endif