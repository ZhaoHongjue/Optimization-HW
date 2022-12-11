#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "Dataset.h"
using namespace std;

Eigen::MatrixXd _get_mat_X(vector<vector<double>> X) {
    int num_example = X.size();
    int num_feature = X[0].size();
    Eigen::MatrixXd X_mat(num_example, num_feature);
    for (int i = 0; i < num_example; i++) {
        for (int j = 0; j < num_feature; j++) {
            X_mat(i, j) = X[i][j];
        }
    }
    return X_mat;
}

Eigen::VectorXd _get_vec_Y(vector<double> Y) {
    int num_example = Y.size();
    Eigen::VectorXd Y_vec(num_example);
    for (int i = 0; i < num_example; i++) {
        Y_vec(i) = Y[i];
    }
    return Y_vec;
}

void Dataset::read_data(string ad) {
    // open the data txt file
    ifstream datafile;
    datafile.open(ad);
    // check whether the data file is open
    if (!datafile.is_open()) 
    {
        cerr << "open data file failed!" << endl;
        return;
    }

    // begin to read the data txt file in "./data/"
    string line;
    vector<vector<double>> X_STL;
    vector<double> Y_STL;
    while (getline(datafile, line)) {
        // get a line from data file
        vector<double> X_line;
        // split data
        istringstream in_line(line);
        string token;
        while (in_line >> token)
        {
            auto pos = token.find(":");
            if (pos != string::npos) {
                // if there is ":" in token, then it is X.
                string number = token.substr(pos + 1);
                X_line.push_back(stod(number));
            } else {
                // else it is Y.
                Y_STL.push_back(stod(token));
            }
        }
        X_STL.push_back(X_line);
    }

    this->X = _get_mat_X(X_STL);
    this->Y = _get_vec_Y(Y_STL);

    // close the data txt file
    if (datafile.is_open())
    {
        datafile.close();
    }
}

Eigen::MatrixXd Dataset::calc_ana_ridge_solution(double lambda) {
    int X_rows = this->X.rows();
    int X_cols = this->X.cols();
    Eigen::MatrixXd X = Eigen::MatrixXd::Constant(X_rows, X_cols + 1, 1);
    X.block(0, 1, X_rows, X_cols) = this->X;
    Eigen::MatrixXd XTX = X.transpose() * X;
    Eigen::MatrixXd temp = (XTX + lambda * Eigen::MatrixXd::Identity(X_cols + 1, X_cols+ 1));
    Eigen::MatrixXd weights = temp.inverse() * X.transpose() * this->Y;
    return weights;
}