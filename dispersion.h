
#ifndef DISPERSION_H_
#define DISPERSION_H_

#include "limmsswarm.h"

#ifdef SIMULATOR
extern float prob_moving;
extern uint8_t base_tumble_time;
extern float offset;
extern float scaling;
extern float d_optim;
extern uint8_t lower_tumble_time;
extern uint16_t upper_tumble_time;

#else
extern float const prob_moving = 0.05;
extern uint8_t const base_tumble_time = 32;
extern float const offset = -5;
extern float const scaling = 64;
extern float const d_optim = 48;
extern uint8_t const lower_tumble_time = 0 * 31;
extern uint16_t const upper_tumble_time = 2 * 32;
#endif

void setup_dispersion();
void start_dispersion();
void behav_dispersion();

#endif

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
