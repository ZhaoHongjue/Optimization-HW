#include <iostream>
#include <eigen3/Eigen/Dense>
#include "Optimizer.h"
#include "Data.h"
#include "LinearModel.h"
#include "utils.h"
#include "Recorder.h"
using namespace std;
using namespace Eigen;

double Optimizer::get_step_size(const VectorXd& d) {
    double num = this->lin.gradient(this->w_decay).dot(d);
    double den = (this->lin.hess(this->w_decay) * d).dot(d);
    return -1 * num / den;
}

/*####################################################################*/
/*                         Gradient Method                            */
/*####################################################################*/

void GradDescent::optimize(Recorder& r, double eps, int mode) {
    int epoch = 0;
    while(1) {
        VectorXd d = -1 * this->lin.gradient(this->w_decay);
        double h = get_step_size(d);
        lin.set_weights(this->lin.get_weights() + h * d);
        double grad_norm = lin.gradient(this->w_decay).norm();
        cout << "epoch: " << epoch << ",\th: " << h << ",\tgrad norm: " << grad_norm << endl;
        r.push_back(this->lin.train_loss(), this->lin.val_loss(), grad_norm);
        if (grad_norm < eps) {
            break;
        }
        epoch += 1;
    }
    r.to_csv();
}

/*####################################################################*/
/*                    Conjugate Gradient Method                       */
/*####################################################################*/

void ConjGrad::optimize(Recorder& r, double eps, int mode) {
    auto w1 = lin.get_weights();
    auto grad1 = lin.gradient(this->w_decay);
    auto d = grad1;
    double beta = 0;
    int epoch = 0;

    while (1) {
        double h = get_step_size(-1 * d);
        lin.set_weights(this->lin.get_weights() - h * d);
        auto grad2 = lin.gradient(this->w_decay);
        cout << "epoch: " << epoch << ",\th: " << h << ",\tgrad norm: " << grad2.norm() << endl;
        r.push_back(this->lin.train_loss(), this->lin.val_loss(), grad2.norm());
        if (grad2.norm() < eps) {
            break;
        }

        if (mode == 0) {
            beta = Dai_Yuan(grad1, grad2, d);
        }
        else if (mode == 1) {
            beta = FR(grad1, grad2);
        }
        else {
            beta = PR(grad1, grad2);
        }

        d = grad2 - beta * d;
        grad1 = grad2;
        epoch += 1;
    }
    r.to_csv();
}

double ConjGrad::Dai_Yuan(const VectorXd& grad1, const VectorXd& grad2, const VectorXd& p) {
    double grad1_norm = grad1.norm();
    double grad2_norm = grad2.norm();
    double num = grad2_norm * grad2_norm;
    double den = (grad2 - grad1).dot(p);
    return num / den;
}
double ConjGrad::FR(const VectorXd& grad1, const VectorXd& grad2) {
    double grad1_norm = grad1.norm();
    double grad2_norm = grad2.norm();
    return -1.0 * (grad2_norm * grad2_norm) / (grad1_norm * grad1_norm);
}
double ConjGrad::PR(const VectorXd& grad1, const VectorXd& grad2) {
    double num = grad2.dot(grad2 - grad1);
    double grad1_norm = grad1.norm();
    double den = grad1_norm * grad1_norm;
    return -1.0 * num / den;
}

/*####################################################################*/
/*                       quasi-Newton Method                          */
/*####################################################################*/

void quasiNetwon::optimize(Recorder& r, double eps, int mode) {
    int epoch = 0;
    int num_feature = this->lin.get_train_X().cols();
    MatrixXd H = MatrixXd::Identity(num_feature + 1, num_feature + 1);
    VectorXd w1 = lin.get_weights();
    VectorXd grad1 = lin.gradient(this->w_decay);

    while (1) {
        VectorXd d = -1 * H * grad1;
        double h = get_step_size(d);
        auto w2 = w1 + h * d;
        lin.set_weights(w2);
        auto grad2 = lin.gradient(this->w_decay);
        cout << "epoch: " << epoch << ",\th: " << h << ",\tgrad norm: " << grad2.norm() << endl;
        r.push_back(this->lin.train_loss(), this->lin.val_loss(), grad2.norm());
        if (grad2.norm() < eps) {break;}

        VectorXd gamma = grad2 - grad1;
        VectorXd delta = w2 - w1;
        if (mode == 0) {
            H = H + Rank1(H, delta, gamma);
        }
        else if (mode == 1) {
            H = H + DFP(H, delta, gamma);
        }
        else {
            H = H + BFGS(H, delta, gamma);
        }

        grad1 = grad2;
        w1 = w2;
        epoch += 1;
    }
    r.to_csv();
}

MatrixXd quasiNetwon::Rank1(const MatrixXd& H, const VectorXd& delta, const VectorXd& gamma) {
    VectorXd term = delta - H * gamma;
    MatrixXd num = term * term.transpose();
    double den = term.dot(gamma);
    return num / den;
}
MatrixXd quasiNetwon::DFP(const MatrixXd& H, const VectorXd& delta, const VectorXd& gamma) {
    MatrixXd term1 = (delta * delta.transpose()) / gamma.dot(delta);
    MatrixXd term2 = (H * gamma * gamma.transpose() * H) / (H * gamma).dot(gamma);
    return term1 - term2;
}
MatrixXd quasiNetwon::BFGS(const MatrixXd& H, const VectorXd& delta, const VectorXd& gamma) {
    double beta = 1 + ((H * gamma).dot(gamma) / gamma.dot(delta));
    MatrixXd term1 = (delta * delta.transpose()) / gamma.dot(delta);
    MatrixXd term2 = (H * gamma * delta.transpose() + delta * gamma.transpose() * H) / gamma.dot(delta);
    return beta * term1 - term2;
}