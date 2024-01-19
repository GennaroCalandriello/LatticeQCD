#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "RunningCoupling.h"
#include "const.h"

using namespace std;

double f(double mu, double alpha, int l) {
  double result = 0;
  vector<double> betas = {beta0, beta1, beta2, beta3};
  for (int i = 0; i < l + 1; i++) {
    result += -(1 / mu) * betas[i] * pow((alpha / (M_PI)), i + 2);
  }
  return result;
}

void rk4(vector<double> &mu_val, vector<double> &alpha_val,
         vector<double> &beta_val, int loop) {
  double k1, k2, k3, k4;
  double alpha_s = 1.;
  double mu0 = mu_start;

  for (int i = 0; i < N; i++) {
    alpha_val.push_back(alpha_s);
    mu_val.push_back(mu0);
    k1 = h * f(mu0, alpha_s, loop);
    k2 = h * f(mu0 + 0.5 * h, alpha_s + 0.5 * k1, loop);
    k3 = h * f(mu0 + 0.5 * h, alpha_s + 0.5 * k2, loop);
    k4 = h * f(mu0 + h, alpha_s + k3, loop);
    alpha_s += (k1 + 2 * k2 + 2 * k3 + k4) / 6;

    mu0 += h;
    // cout << "mu0 = " << mu0 << endl;
  }
}

void RungeKuttaFehldberg(vector<double> &mu_val, vector<double> &alpha_val,
                         vector<double> &beta_val, int loop) {
  double k1, k2, k3, k4, k5;
  double alpha_s = 1.;
  double mu0 = mu_start;

  for (int i = 0; i < N; i++) {
    k1 = h * f(mu0, alpha_s, loop);
    k2 = h * f(mu0 + a2 * h, alpha_s + b21 * k1, loop);
    k3 = h * f(mu0 + a3 * h, alpha_s + b31 * k1 + b32 * k2, loop);
    k4 = h * f(mu0 + a4 * h, alpha_s + b41 * k1 + b42 * k2 + b43 * k3, loop);
    k5 = h * f(mu0 + a5 * h,
               alpha_s + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, loop);

    // wait(2);
    double alpha_s_new =
        alpha_s + c1 * k1 + c2 * k2 + c3 * k3 + c4 * k4 + c5 * k5;
    double alpha_s_err = ct1 * k1 + ct2 * k2 + ct3 * k3 + ct4 * k4 + ct5 * k5;
    double hnew = 0.9 * h * pow(abs(eps / alpha_s_err), 0.2);

    // if (abs(alpha_s_err) > eps) {

    // cout << "k1 = " << k1 << endl;
    // cout << "k2 = " << k2 << endl;
    // cout << "k3 = " << k3 << endl;
    // cout << "k4 = " << k4 << endl;
    // cout << "k5 = " << k5 << endl;
    // cout << "alpha_s = " << alpha_s << endl;
    // cout << "alpha_s_new = " << alpha_s_new << endl;
    // cout << "alpha_s_err = " << alpha_s_err << endl;
    // cout << "h = " << h << endl;
    // cout << "hnew = " << hnew << endl;
    // cout << "mu0 = " << mu0 << endl;

    alpha_s += alpha_s_new;
    mu0 += h;
    h = hnew;
    alpha_val.push_back(alpha_s);
    mu_val.push_back(mu0);
    // } else {

    //   continue;
    // }
  }
}

// Function to pause execution for a specified number of seconds
void wait(int seconds) {
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

// double CRunDec::fRungeKuttaImpl(double &x, double y, double &htry, int nl,
//                               double (*f)(CRunDec, double, int)){
//      // Precision
//      double eps=1e-10;
//      double yerr,ytemp,htemp, hnext;
//      double h=htry;
//      double xnew; // new variable
//      double k1,k2,k3,k4,k5,k6;
//      for(;;){
//        k1=h*f(*this,y,nl);
//        k2=h*f(*this,y+b21*k1,nl);
//        k3=h*f(*this,y+b31*k1+b32*k2,nl);
//        k4=h*f(*this,y+b41*k1+b42*k2+b43*k3,nl);
//        k5=h*f(*this,y+b51*k1+b52*k2+b53*k3+b54*k4,nl);
//        k6=h*f(*this,y+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5,nl);
//        // y value at x+h as a sum of the previous value and the
//        // correspondingly weighted function evaluations
//        ytemp= y+ c1*k1+ c2*k2+ c3*k3+ c4*k4+ c5*k5+ c6*k6;
//        // Estimate of uncertainty
//        yerr=dc1*k1 + dc2*k2 + dc3*k3 + dc4*k4 + dc5*k5 + dc6*k6;
//        double err=0.;
//        err=fmax(err,fabs(yerr/eps));

//        // Uncertainty too big? -> Discard result and reduce step size
//        if(err>1.){
//          htemp=0.9*h*pow(err,-0.25);
//          if(h>=0.){h=fmax(htemp,0.1*h);}
//          else{h=fmin(htemp,0.1*h);}
//          xnew=x+h;  // modification to previous code
//          //decide whether reduced stepsize is still big enough
//          //(in order to prevent a closed loop)
//          if(xnew==x){cout<<"stepsize too small"<<endl; RETURN}
//          continue;
//        }
//        else{
//          if(err>1.89e-4){
//          hnext=0.9*h*pow(err,-0.2);
//          }
//          // Uncertainty OK? -> take y value, increase h
//          else{
//            hnext=5.*h;
//          }
//          x+=h;

//          y=ytemp;
//          htry=hnext;
//          break;
//        }
//      }
//      return y;
// }

void writeToFile(const std::vector<double> &mu_values,
                 const std::string &filename) {
  std::ofstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Impossibile aprire il file: " << filename << std::endl;
    return;
  }

  for (size_t i = 0; i < mu_values.size(); ++i) {
    file << mu_values[i] << std::endl;
  }

  file.close();
}

void Integrator() {
  // Parameters
  // step size
  for (int i = 0; i < loops + 1; i++) {
    // Arrays to store the values
    std::vector<double> mu_values, alpha_values, beta_values;

    // Solve the system
    // rk4(mu_values, alpha_values, beta_values, i);
    RungeKuttaFehldberg(mu_values, alpha_values, beta_values, i);
    cout << "mu_values.size() = " << mu_values.size() << endl;
    std::string file_a1 = "alpha_" + std::to_string(i) + ".txt";
    std::string file_mu1 = "mu_" + std::to_string(i) + ".txt";

    writeToFile(mu_values, file_mu1);
    writeToFile(alpha_values, file_a1);
  }
}

int main() {
  Integrator();
  // Output the results
  cout << "betas" << beta0 << " " << beta1 << " " << beta2 << " " << beta3
       << endl;

  return 0;
}