#include <iostream>
#include <eigen3/Eigen/Dense>
#include "utils.h"
using namespace std;
using namespace Eigen;

MatrixXd _get_extended_X(const MatrixXd& X) {
    int X_rows = X.rows();
    int X_cols = X.cols();
    MatrixXd X_extended = MatrixXd::Constant(X_rows, X_cols + 1, 1);
    X_extended.block(0, 1, X_rows, X_cols) = X;
    return X_extended;
}

double _calc_loss(const VectorXd& Y, const VectorXd& Y_hat) {
    VectorXd err = Y - Y_hat;
    return (err.transpose() * err)(0, 0) / err.size();
}

VectorXd _get_gradient(const MatrixXd& X, const VectorXd& Y, const VectorXd& weights, double lambda) {
    int num_example = X.rows();
    auto X_extended = _get_extended_X(X);
    auto pMSE = -2.0 / num_example * X_extended.transpose() * (Y - X_extended * weights);
    auto pPenalty = 2 * lambda * weights;
    return pMSE + pPenalty;
}
