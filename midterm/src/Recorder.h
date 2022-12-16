#ifndef _RECORDER_H_
#define _RECORDER_H_

#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

class Recorder {
private:
    vector<double> train_losses;
    vector<double> val_losses;
    vector<double> grad_norms;
    string ad;
public:
    Recorder(string ad): ad(ad) {};
    void push_back(double train_loss, double val_loss, double grad_norm);
    void to_csv();
};

#endif