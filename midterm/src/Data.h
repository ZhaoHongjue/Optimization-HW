#ifndef _DATA_H_
#define _DATA_H_

#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>
using namespace std;
using namespace Eigen;

class Data {
private:
    MatrixXd X;
    VectorXd Y;
public:
    void read_data(string ad);
    MatrixXd calc_ana_solution(double lambda);
    int get_feature_num() {return X.cols();};
    MatrixXd get_X() {return this->X;};
    VectorXd get_Y() {return this->Y;};
    void print_X() {cout << this->X << endl;};
    void print_Y() {cout << this->Y << endl;};
};

#endif