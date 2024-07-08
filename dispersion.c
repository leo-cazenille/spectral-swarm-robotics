/* TODO
 */

#include <kilombo.h>
#include <math.h>
#include "dispersion.h"



#ifdef SIMULATOR
float prob_moving = 0.05;
uint8_t base_tumble_time = 32;
float offset = -5;
float scaling = 64;
float d_optim = 48;
uint8_t lower_tumble_time = 0 * 31;
uint16_t upper_tumble_time = 4 * 31;

#else
float const prob_moving = 0.05;
uint8_t const base_tumble_time = 32;
float const offset = -5;
float const scaling = 64;
float const d_optim = 48;
uint8_t const lower_tumble_time = 0 * 31;
uint16_t const upper_tumble_time = 2 * 32;
#endif

uint8_t const dist_with_no_neighbors = 255; // big enough



float uniform(void) {
    return ((float)rand()+1.0)/((float)RAND_MAX+2.0);
}

float rand_normal(float mu, float sigma) {
    float z = sqrt(-2.0*log(uniform())) * sin(2.0*M_PI*uniform());
    return mu + sigma*z;
}

void get_d_min() {
    int8_t i;
    uint8_t candidate_d_min = dist_with_no_neighbors;

    for(i = 0; i < mydata->nb_neighbors; i++) {
        if (candidate_d_min > mydata->neighbors[i].dist)
            candidate_d_min = mydata->neighbors[i].dist;
    }
    mydata->d_min = candidate_d_min;
}

void get_d_max() {
    int8_t i;
    uint8_t candidate_d_max = 0;

    for(i = 0; i < mydata->nb_neighbors; i++) {
        if (candidate_d_max < mydata->neighbors[i].dist)
            candidate_d_max = mydata->neighbors[i].dist;
    }
    mydata->d_max = candidate_d_max;
}

void setup_dispersion() {
    for(;;) {
        mydata->tumble_time = fabs(base_tumble_time + rand_normal(0, 1) * 32); // 2 sec // not too big
        if (mydata->tumble_time < upper_tumble_time && mydata->tumble_time > lower_tumble_time) break;
    }
    mydata->run_time = 64; // 255;
    mydata->direction = rand_soft() % 2;
    mydata->prob = ((float)rand_soft() / 255.0f) * ((float)rand_soft() / 255.0f);
}

void start_dispersion() {
    mydata->last_kiloticks_dispersion = kilo_ticks;
}

void behav_dispersion() {
    /*
    run and tumble algorithm
    */
    get_d_min();
    get_d_max();
    
    mydata->cycle_dispersion = kilo_ticks - mydata->last_kiloticks_dispersion;

    if (mydata->flag_dispersion == 0) {
        for(;;) {
            mydata->tumble_time = fabs(base_tumble_time + rand_normal(0, 1) * 32); // 2 sec // not too big
            if (mydata->tumble_time < upper_tumble_time && mydata->tumble_time > lower_tumble_time) break;
        }

        float const is_positive_frustration = 1.0f - mydata->d_min / d_optim;
        if (is_positive_frustration > 0) // d_min < d_optim
            mydata->frustration = is_positive_frustration;
        else if (mydata->d_min < dist_with_no_neighbors) // if it has some neighbors, it doesn't have to move // d_min >= d_optim
            mydata->frustration = 0;
        else // if it has no neighbors, it needs to explore // d_min == dist_with_no_neighbors
            mydata->frustration = is_positive_frustration * -1.0f;

        float const is_positive_run_time = (float)offset + (float)mydata->frustration * (float)scaling;
        if (is_positive_run_time > 0)
            mydata->run_time = is_positive_run_time;
        else
            mydata->run_time = 0;

        mydata->direction = rand_soft() % 2;
        mydata->prob = ((float)rand_soft() / 255.0f) * ((float)rand_soft() / 255.0f);
        mydata->flag_dispersion = 1;

    } else if (mydata->prob < prob_moving) { // move
        if (mydata->cycle_dispersion < mydata->tumble_time) {
            // tumble state
            spinup_motors();
            set_color(RGB(3,0,0)); // red
            if(mydata->direction)
                set_motors(kilo_turn_right, 0);
            else
                set_motors(0, kilo_turn_left);
        } else if (mydata->cycle_dispersion < mydata->tumble_time + mydata->run_time) {
            // run state
            spinup_motors();
            set_motors(kilo_straight_left, kilo_straight_right);
            set_color(RGB(0,3,0)); // green
        } else {
            mydata->last_kiloticks_dispersion = kilo_ticks;
            mydata->flag_dispersion = 0;
        }

    } else { // stop
        if (mydata->cycle_dispersion < mydata->tumble_time + mydata->run_time) {
            set_motors(0, 0);
            set_color(RGB(3,3,3)); // white
        } else {
            mydata->last_kiloticks_dispersion = kilo_ticks;
            mydata->flag_dispersion = 0;
        }
    }

//     set_color_from_nb_neighbours();
}



// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
