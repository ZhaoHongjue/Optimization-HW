#ifndef _LINEAR_MODEL_H
#define _LINEAR_MODEL_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "Data.h"
#include "utils.h"
using namespace std;
using namespace Eigen;

class LinearModel {
private:
    VectorXd weights;
    Data data;
public:
    LinearModel(const Data& data);
    LinearModel(const Data& data, const VectorXd& weights): data(data), weights(weights) {};
    ~LinearModel() {};

    VectorXd forward(const MatrixXd& X);
    VectorXd forward_train() {return forward(get_train_X());};
    VectorXd forward_val() {return forward(get_val_X());};

    VectorXd gradient(double lambda);
    MatrixXd hess(double lambda);

    MatrixXd get_train_X() {return this->data.get_train_X();};
    VectorXd get_train_Y() {return this->data.get_train_Y();};
    MatrixXd get_val_X() {return this->data.get_val_X();};
    VectorXd get_val_Y() {return this->data.get_val_Y();};
    double train_loss() {return _calc_loss(get_train_Y(), forward_train());};
    double val_loss() {return _calc_loss(get_val_Y(), forward_val());};
    
    VectorXd get_weights() {return this->weights;};
    void set_weights(VectorXd weights) {this->weights = weights;};
    void set_ideal_weights(double w_decay) {
        this->weights = data.calc_ana_solution(w_decay);
    };

    void set_data(const Data& data) {
        this->data = data;
    }
};

#endif