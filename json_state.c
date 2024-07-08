/* Saving bot state as json. Not for use in the real bot, only in the simulator. */
#include <kilombo.h>

#ifdef SIMULATOR

#include <jansson.h>
#include <stdio.h>
#include <string.h>
#include "limmsswarm.h"

json_t *json_state()
{
    // Create the state object we return
    json_t* state = json_object();

    // Store all relevant values
    // Note: make sure there are no NaN values, as they will not be saved at all by jansson !
    json_object_set_new(state, "current_behavior", json_integer(mydata->current_behavior));
    //json_object_set_new(state, "diffusion_valid", json_boolean(mydata->curr_diff->diffusion_valid));
    json_object_set_new(state, "diffusion_valid1", json_boolean(mydata->diff1.diffusion_valid));
    json_object_set_new(state, "t", json_real(mydata->curr_diff->t));
    json_object_set_new(state, "nb_neighbors", json_real(mydata->nb_neighbors));

    json_object_set_new(state, "s", json_real(mydata->curr_diff->s[0]));
    json_object_set_new(state, "lambda", json_real(mydata->diff1.lambda));
    json_object_set_new(state, "avg_lambda", json_real(mydata->diff1.avg_lambda));

    json_object_set_new(state, "last_mse_0", json_real(mydata->diff1.last_mse[0]));
#if defined(ENABLE_DOUBLE_DIFF)
    json_object_set_new(state, "last_mse_1", json_real(mydata->diff1.last_mse[1]));
    json_object_set_new(state, "last_mse_2", json_real(-1.f));
#elif defined(ENABLE_TRIPLE_DIFF)
    json_object_set_new(state, "last_mse_1", json_real(mydata->diff1.last_mse[1]));
    json_object_set_new(state, "last_mse_2", json_real(mydata->diff1.last_mse[2]));
#else
    json_object_set_new(state, "last_mse_1", json_real(-1.f));
    json_object_set_new(state, "last_mse_2", json_real(-1.f));
#endif
    return state;
}

#endif
// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
