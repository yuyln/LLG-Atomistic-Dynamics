#ifndef __RANDOM_H
#define __RANDOM_H

double nsrandom(tyche_i_state *state, double start, double end);
double normal_distribution_box_muller(tyche_i_state *state);
double normal_distribution(tyche_i_state *state);
double get_random_gsa_(tyche_i_state *state, double qV, double T, double gamma);
double get_random_gsa(tyche_i_state *state, double qV, double T, double _gamma);

#endif
