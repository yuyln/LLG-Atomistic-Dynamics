#include "random.h"

double nsrandom(tyche_i_state *state, double start, double end) {
    return tyche_i_double((*state)) * (end - start) + start;
}

//@TODO Proper normal
double normal_distribution_box_muller(tyche_i_state *state) {
    double u1 = nsrandom(state, 0, 1);
    double u2 = nsrandom(state, 0, 1);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

double normal_distribution(tyche_i_state *state) {
    for (;;) {
        double U = nsrandom(state, 0, 1);
        double V = nsrandom(state, 0, 1);
        double X = sqrt(8.0 / M_E) * (V - 0.5) / U;
        double X2 = X * X;
        if (X2 <= (5.0 - 4.0 * exp(0.25) * U))
            return X;
        else if (X2 >= (4.0 * exp(-1.35) / U + 1.4))
            continue;
        else if (X2 <= (-4.0 * log(U)))
            return X;
    }
}

//Assume D=1
double get_random_gsa_(tyche_i_state *state, double qV, double T, double gamma) {
    double dx = nsrandom(state, -10, 10);
    double c = sqrt((qV - 1.0) / M_PI) * gamma * pow(T, -1.0 / (3.0 - qV));
    double l = pow(1.0 + (qV - 1) * dx * dx / pow(T, 2.0 / (3.0 - qV)), 1.0 / (qV - 1.0));
    return c * dx / l;
}

double get_random_gsa(tyche_i_state *state, double qV, double T, double gamma_) {
    double f1 = exp(log(T) / (qV - 1.0));
    double f2 = exp((4.0 - qV) * log(qV - 1.0));
    double f3 = exp((2.0 - qV) * log(2.0) / (qV - 1.0));
    double f4 = sqrt(M_PI) * f1 * f2 / (f3 * (3.0 - qV));
    double f5 = 1.0 / (qV - 1.0) - 0.5;
    double f6 = M_PI * (1.0 - f5) / sin(M_PI * (1.0 - f5)) / tgamma(2.0 - f5);
    double sigmax = exp(-(qV - 1.0) * log(f6 / f4) / (3.0 - qV));
    double x = sigmax * normal_distribution(state);
    double y = normal_distribution(state);
    double den = exp((qV - 1.0) * log(fabs(y)) / (3.0 - qV));
    return x / den;
}
