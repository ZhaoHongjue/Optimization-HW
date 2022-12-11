#ifndef _DATASET_H_
#define _DATASET_H_

#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>
using namespace std;

class Dataset {
private:
    Eigen::MatrixXd X;
    Eigen::VectorXd Y;
public:
    void read_data(string ad);
    Eigen::MatrixXd calc_ana_ridge_solution(double lambda);
    int get_feature_num() {return X.cols();};
    Eigen::MatrixXd get_X() {return this->X;};
    Eigen::VectorXd get_Y() {return this->Y;};
    void print_X() {cout << this->X << endl;};
    void print_Y() {cout << this->Y << endl;};
};

#endif