#include <cmath>

double nf = 6.;
double h = 0.001; // integration step
double mu_end = 10.;
double mu_start = 1.;
double mu0 = mu_start;
int N = (mu_end - mu_start) / h;
int loops = 3;
double xi3 = 1.202057;
double beta0 = 0.25 * (11. - 2. / 3. * nf);
double beta1 = (1. / 16.) * (102. - 38. / 3. * nf);
double beta2 =
    (1. / 64.) * (2857. / 2 - 5033. / 18. * nf + 325. / 54. * pow(nf, 2));
double beta3 = (1. / 256.) * (149753. / 6. + 3564. * pow(xi3, 2) -
                              1078361. / 162. * nf - 6508. / 27. * xi3 * nf +
                              (50065. / 162. + 6472. / 81. * xi3) * pow(nf, 2) +
                              1093. / 729. * pow(nf, 3));

// double b21 = 2. / 9., b31 = 1. / 12., b32 = 1. / 4., b41 = 69. / 128.,
//        b42 = -243. / 128., b43 = 135. / 64., b51 = -17. / 12., b52 = 27.
//        / 4., b53 = -27. / 5., b54 = 16. / 15.;

// double a2 = 2. / 9., a3 = 1. / 3., a4 = 3.4, a5 = 1., a6 = 5. / 6.;
// double c1 = 47. / 450., c2 = 0., c3 = 12 / 25, c4 = 32 / 225, c5 = 1. / 20.;
// double ct1 = 1. / 150., ct2 = 0., ct3 = -3. / 100., ct4 = 16. / 75.,
//        ct5 = 1. / 20.;
double eps = 1e-5;

const double b21 = 1. / 5., b31 = 0, b41 = 9. / 4., b51 = -63. / 100.,
             b32 = 2. / 5., b42 = -5, b52 = 9. / 5., b43 = 15. / 4.,
             b53 = -13. / 20., b54 = 2. / 25.;

const double ct1 = -0.139 * 1e-2, ct2 = -0.556 * 1e-3, ct3 = -0.833 * 1e-5,
             ct4 = -0.278 * 1e-3, ct5 = 0.278 * 1e-2;

const double c1 = 17.0 / 144.0, c2 = 0.0, c3 = 25.0 / 36.0, c4 = 1.0 / 72.0,
             c5 = -25.0 / 72.0, c6 = 25.0 / 48.0;
const double a2 = 1.0 / 5.0, a3 = 2. / 5., a4 = 1., a5 = 3. / 5., a6 = 4. / 5.;