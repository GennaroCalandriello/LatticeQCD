#ifndef RC_H
#define RC_H

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
double f(double alpha, double mu, int l);
void rk4(vector<double> &mu_val, vector<double> &alpha_val,
         vector<double> &beta_val, int l);
void RungeKuttaFehldberg(vector<double> &mu_val, vector<double> &alpha_val,
                         vector<double> &beta_val, int loop);
void wait(int seconds);
void Integrator();
#endif // RC_H