#include <iostream>
#include <eigen3/Eigen/Dense>
#include "utils.h"
#include "LinearModel.h"
using namespace std;
using namespace Eigen;

LinearModel::LinearModel(int num_feature) {
    this->weights = MatrixXd::Zero(num_feature + 1, 1);
}

LinearModel::LinearModel(const VectorXd& ideal_weights) {
    this->weights = ideal_weights;
}

VectorXd LinearModel::forward(const MatrixXd& X) {
    return _get_extended_X(X) * this->weights;
}
