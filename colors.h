
#ifndef COLORS_H_
#define COLORS_H_

#include "limmsswarm.h"

#ifdef SPECIFY_CENTROIDS_IN_CONFIG
extern float class_centroids[];
#endif

void set_color_from_lambda(ext_t lambda);
void set_color_from_s(ext_t x);
void set_color_from_signs(ext_t x);
void set_color_from_nb_neighbours();

#endif

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
