#include <iostream>
#include <eigen3/Eigen/Dense>
#include "Data.h"
#include "utils.h"
#include "LinearModel.h"
using namespace std;
using namespace Eigen;

LinearModel::LinearModel(const Data& data): data(data) {
    // this->weights = MatrixXd::Constant(this->data.get_feature_num() + 1, 1, 0.1);
    this->weights = MatrixXd::Random(this->data.get_feature_num() + 1, 1);
}

VectorXd LinearModel::gradient(double lambda) {
    auto X = this->data.get_train_X();
    auto Y = this->data.get_train_Y();
    auto X_extended = _get_extended_X(X);
    auto pMSE = -2.0 * X_extended.transpose() * (Y - X_extended * weights);
    auto pPenalty = 2 * lambda * weights;
    return pMSE + pPenalty;
}

MatrixXd LinearModel::hess(double lambda) {
    auto X = this->data.get_train_X();
    MatrixXd X_extended = _get_extended_X(X);
    int cols = X_extended.cols();
    MatrixXd XTX = X_extended.transpose() * X_extended;
    return 2 * (XTX + lambda * MatrixXd::Identity(cols, cols));
}

VectorXd LinearModel::forward(const MatrixXd& X) {
    return _get_extended_X(X) * this->weights;
}
