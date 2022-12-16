#ifndef _DATA_H_
#define _DATA_H_

#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>
using namespace std;
using namespace Eigen;

class Data {
private:
    MatrixXd total_X;
    VectorXd total_Y;
    double split_rate;
public:
    Data(double split_rate): split_rate(split_rate) {};
    void read_data(string ad);
    MatrixXd calc_ana_solution(double lambda);

    int get_feature_num() {return this->total_X.cols();};
    MatrixXd get_train_X();
    VectorXd get_train_Y();
    MatrixXd get_val_X();
    VectorXd get_val_Y();
};

#endif