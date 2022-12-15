#include <iostream>
#include <eigen3/Eigen/Dense>
#include "Optimizer.h"
#include "Data.h"
#include "utils.h"
#include <cmath>
using namespace std;
using namespace Eigen;

void GradDescent::optimize(LinearModel& lin, Data data, double eps) {
    auto X = data.get_X();
    auto Y = data.get_Y();
    auto w1 = lin.get_weights();
    auto grad1 = _get_gradient(X, Y, w1, this->lambda);
    int epoch = 0;

    while (epoch < 100) {
        auto w2 = w1 - (this->lr / sqrt(epoch + 1))  * grad1;
        auto grad2 = _get_gradient(X, Y, w2, this->lambda);
        lin.set_weights(w2);
        auto Y_hat = lin.forward(X);
        double loss = _calc_loss(Y_hat, Y);
        cout << "epoch: " << epoch <<",\tgrad norm: " << grad2.norm() << ",\tloss: " << loss << endl;
        if (grad2.norm() < eps) {
            lin.set_weights(w2);
            break;
        }
        w1 = w2;
        grad1 = grad2;
        epoch += 1;
    }

    // auto X = data.get_X();
    // auto Y = data.get_Y();
    // auto last_Y_hat = lin.forward(X);
    // int epoch = 0;
    // while (epoch < 100) {
    //     auto w1 = lin.get_weights();
    //     auto grad = _get_gradient(X, Y, w1, this->lambda);
    //     if (grad.norm() < eps) {break;}
    //     auto w2 = w1 - this->lr * grad;
    //     lin.set_weights(w2);
    //     auto Y_hat = lin.forward(X);
    //     auto loss = _calc_loss(Y, Y_hat);
    //     cout << "epoch: " << epoch << ",\tloss: " << loss << ",\tgrad norm:" <<grad.norm() << endl; 
    //     if ((Y_hat - last_Y_hat).norm() < eps || (w2 - w1).norm() < eps) {break;}
    //     last_Y_hat = Y_hat;
    //     epoch += 1;
    // }
}
/*####################################################################*/
/*                    Conjugate Gradient Method                       */
/*####################################################################*/

void ConjGrad::optimize(LinearModel& lin, Data data, double eps, int mode) {
    auto X = data.get_X();
    auto Y = data.get_Y();
    auto w1 = lin.get_weights();
    auto grad1 = _get_gradient(X, Y, w1, this->lambda);
    auto p = grad1;
    double beta = 0;
    int epoch = 0;

    while (epoch < 100) {
        auto w2 = w1 - this->lr * p;
        auto grad2 = _get_gradient(X, Y, w2, this->lambda);
        cout << "epoch: " << epoch <<",\tgrad norm: " << grad2.norm() << endl;
        if (grad2.norm() < eps) {
            lin.set_weights(w2);
            break;
        }

        if (mode == 0) {beta = Dai_Yuan(grad1, grad2, p);}
        else if (mode == 1) {beta = FR(grad1, grad2);}
        else {beta = PR(grad1, grad2);}
        p = grad2 - beta * p;
        w1 = w2;
        grad1 = grad2;
        epoch += 1;
    }
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

void quasiNetwon::optimize(LinearModel& lin, Data data, double eps, int mode) {
    auto X = data.get_X();
    auto Y = data.get_Y();
    auto w1 = lin.get_weights();
    auto grad1 = _get_gradient(X, Y, w1, this->lambda);
    MatrixXd H = MatrixXd::Identity(w1.rows(), w1.rows());
    int epoch = 0;
    while (epoch < 100) {
        auto p = H * grad1;
        auto w2 = w1 - this->lr * p;
        auto grad2 = _get_gradient(X, Y, w2, this->lambda);
        cout << "epoch: " << epoch <<",\tgrad norm: " << grad2.norm() << endl;
        if (grad2.norm() < eps) {
            lin.set_weights(w2);
            break;
        }

        VectorXd gamma = grad2 - grad1;
        VectorXd delta = w2 - w1;

        if (mode == 0) {H = Rank1(H, delta, gamma);}
        else if (mode == 1) {H = DFP(H, delta, gamma);}
        else {H = BFGS(H, delta, gamma);}

        w1 = w2;
        grad1 = grad2;
        epoch += 1;
    }
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