/* TODO
 */

#include <kilombo.h>
#include <stdbool.h>
#include <math.h>

#ifdef SIMULATOR
#include <limits.h>
#else
#include <avr/io.h>
#include <stdlib.h>
#endif

#include "util.h"
#include "colors.h"
#include "limmsswarm.h"  // defines the USERDATA structure
#include "simulation.h"
#include "dispersion.h"
REGISTER_USERDATA(USERDATA)



#ifdef SIMULATOR
float initial_s_max_val = 1.f;
float inv_tau = 20.f;

float diffusion_convergence_threshold = 0.1f;
uint16_t diffusion_min_nb_points = 8;
float diffusion_min_abs_s = 0.1f;

uint16_t rnd_comms_radius = 0;
uint16_t rnd_noise_dist = 0;
uint16_t rnd_noise_dist_each_it = 0;

uint32_t kiloticks_initial_random_walk = 16000;
uint32_t kiloticks_random_walk_choice = 100;
uint32_t kiloticks_randow_walk = 4560 * 1.0; // 4096;
uint32_t kiloticks_handshake = 6200;
uint32_t kiloticks_diffusion = 2480 * 2.0; // 2480;
uint32_t kiloticks_diffusion_it = 62;
uint32_t kiloticks_diffusion_burnin = 2480 * 1.0;
uint32_t kiloticks_collective_avg_lambda = 2480 * 2.0; // 4960; //2480; //; //1240;
uint32_t kiloticks_collective_avg_lambda_it = 62;
uint32_t kiloticks_collective_avg_avg_lambda = 2480 * 2.0; // 4960; // 2480; //4960; //1240;
uint32_t kiloticks_collective_avg_avg_lambda_it = 62;

uint32_t kiloticks_start_it_waiting_time = 930; // 3720; // 2790; // 1860;
uint32_t kiloticks_end_it_waiting_time = 0;

#else
// REAL ROBOTS

float const initial_s_max_val = 1.f;
float const inv_tau = 50.f;

float const diffusion_convergence_threshold = 0.1f;
uint16_t const diffusion_min_nb_points = 8;
float const diffusion_min_abs_s = 0.e-05f;

uint32_t const kiloticks_initial_random_walk = 0;
uint32_t const kiloticks_random_walk_choice = 15;
uint32_t const kiloticks_randow_walk = 0;
uint32_t const kiloticks_handshake = 6200;
uint32_t const kiloticks_diffusion = 54250;
uint32_t const kiloticks_diffusion_it = 248;
uint32_t const kiloticks_diffusion_burnin = 10000;
uint32_t const kiloticks_collective_avg_lambda = 15500;
uint32_t const kiloticks_collective_avg_lambda_it = 248;
uint32_t const kiloticks_collective_avg_avg_lambda = 15500;
uint32_t const kiloticks_collective_avg_avg_lambda_it = 248;

uint32_t const kiloticks_start_it_waiting_time = 930;
uint32_t const kiloticks_end_it_waiting_time = 0;

#endif

uint32_t kiloticks_iteration = 0; // Set in ``setup()``


#ifdef SIMULATOR
#include <jansson.h>
json_t *json_state();

void init_params() {
    init_float(initial_s_max_val);
    init_float(inv_tau);
    init_int(rnd_comms_radius);
    init_int(rnd_noise_dist);
    init_int(rnd_noise_dist_each_it);

    init_float(diffusion_convergence_threshold);
    init_int(diffusion_min_nb_points);
    init_float(diffusion_min_abs_s);

    init_int(kiloticks_initial_random_walk);
    init_int(kiloticks_random_walk_choice);
    init_int(kiloticks_randow_walk);
    init_int(kiloticks_handshake);
    init_int(kiloticks_diffusion);
    init_int(kiloticks_diffusion_it);
    init_int(kiloticks_diffusion_burnin);
    init_int(kiloticks_collective_avg_lambda);
    init_int(kiloticks_collective_avg_lambda_it);
    init_int(kiloticks_collective_avg_avg_lambda);
    init_int(kiloticks_collective_avg_avg_lambda_it);

    init_int(kiloticks_end_it_waiting_time);
    init_int(kiloticks_start_it_waiting_time);

    // Dispersion
    init_float(prob_moving);
    init_int(base_tumble_time);
    init_float(offset);
    init_float(scaling);
    init_float(d_optim);
    init_float(lower_tumble_time);
    init_float(upper_tumble_time);

#ifdef SPECIFY_CENTROIDS_IN_CONFIG
#define init_float(X) {X = get_float_param(#X , X); }
    // Centroids, if needed
    class_centroids[0] = get_float_param("centroid_error", class_centroids[0]);     // Error
    class_centroids[1] = get_float_param("centroid_purple0", class_centroids[1]);   // Annulus
    class_centroids[2] = get_float_param("centroid_purple1", class_centroids[2]);   // Annulus
    class_centroids[3] = get_float_param("centroid_cyan", class_centroids[3]);      // Disk
    class_centroids[4] = get_float_param("centroid_orange", class_centroids[4]);    // Square
    class_centroids[5] = get_float_param("centroid_green", class_centroids[5]);     // Arrow2
    class_centroids[6] = get_float_param("centroid_red", class_centroids[6]);       // Star
    class_centroids[7] = get_float_param("centroid_brown", class_centroids[7]);     // 8
    class_centroids[8] = get_float_param("centroid_grey", class_centroids[8]);      // 6
    class_centroids[9] = get_float_param("centroid_yellow", class_centroids[9]);    // Triangle
#endif
}
#endif


bool is_number_valid(ext_t nb) {
    return !isnan(nb) && !isinf(nb);
}



#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
inline uint32_t to_uint(float const x) {
    return *(uint32_t*)&x;
}
inline float to_float(uint32_t const x) {
    return *(float*)&x;
}
#pragma GCC diagnostic pop


#if defined(ENABLE_DOUBLE_DIFF)
inline float i24bits_to_float(uint32_t const x) {
    return ((float)x) * 500.f / 8388608.f - 250.f;
}

uint32_t float_to_24bits(float const x) {
    float x2 = x;
    if(x2 < -250.f)
        x2 = -250.f;
    if(x2 > 250.f)
        x2 = 250.f;
    x2 = (x2+250.f) * 8388608.f / 500.f; // 2**23 = 8388608
    return roundf(x2);
}
#elif defined(ENABLE_TRIPLE_DIFF)
// From https://stackoverflow.com/a/60047308
float half_to_float(uint16_t const x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    uint32_t const x2 = x;
    uint32_t const e = (x2&0x7C00)>>10; // exponent
    uint32_t const m = (x2&0x03FF)<<13; // mantissa
    uint32_t const v = to_uint((float)m)>>23; // evil log2 bit hack to count leading zeros in denormalized format
    return to_float((x2&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FE000))); // sign : normalized : denormalized
}

// From https://stackoverflow.com/a/60047308
uint16_t float_to_half(float const x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    uint32_t const b = to_uint(x)+0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
    uint32_t const e = (b&0x7F800000)>>23; // exponent
    uint32_t const m = b&0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    return (b&0x80000000)>>16 | (e>112)*((((e-112)<<10)&0x7C00)|m>>13) | ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) | (e>143)*0x7FFF; // sign : normalized : denormalized : saturate
}

#elif defined(ENABLE_HEXA_DIFF)
float i8bits_to_float(uint8_t const x) {
    return ((float)x) * 10.f / 128.f - 5.f;
}

uint8_t float_to_i8bits(float const x) {
    float x2 = x;
    if(x2 < -10.f)
        x2 = -10.f;
    if(x2 > 10.f)
        x2 = 10.f;
    x2 = (x2+5.f) * 128.f / 10.f;
    return roundf(x2);
}
#endif



/* Go through the list of neighbors, remove entries older than a threshold,
 * currently 2 seconds.
 */
void purgeNeighbors(void) {
    int8_t i;
    for (i = mydata->nb_neighbors-1; i >= 0; i--)
        if (kilo_ticks - mydata->neighbors[i].timestamp > kiloticks_diffusion_it) { // 31 ticks = 1 s
            // This one is too old.
            mydata->neighbors[i] = mydata->neighbors[mydata->nb_neighbors-1]; // Replace it by the last entry
            mydata->nb_neighbors--;
        }
}


void clearNeighbors(void) {
    mydata->nb_neighbors = 0;
}


#ifdef ENABLE_HANDSHAKES

void clearKnownNeighbors(void) {
    mydata->nb_known_neighbors = 0;
    mydata->current_peer_index = 0;
}

bool is_neighbor_known(uint16_t uid) {
    for(int8_t i = mydata->nb_known_neighbors-1; i >= 0; i--) {
        if(uid == mydata->known_neighbors_uid[i])
            return true;
    }
    return false;
}

#endif




void setup_message(void) {
    uint8_t index = 0;
    //mydata->msg.type = NORMAL;
    mydata->msg.type = (int8_t)mydata->data_type;

    mydata->msg.data[index++] = kilo_uid & 0xff;         // 0 low  ID
    mydata->msg.data[index++] = kilo_uid >> 8;           // 1 high ID

    if(mydata->data_type == DATA_S) {
#if defined(ENABLE_SINGLE_DIFF)
        int32_t casted_val = to_uint(mydata->val[0]);
        mydata->msg.data[index++] = ((int32_t)casted_val) & 0xFF;
        mydata->msg.data[index++] = (((int32_t)casted_val)>>8)&0xFF;
        mydata->msg.data[index++] = (((int32_t)casted_val)>>16)&0xFF;
        mydata->msg.data[index++] = (((int32_t)casted_val)>>24)&0xFF;
#elif defined(ENABLE_DOUBLE_DIFF)
        int32_t const half1 = float_to_24bits(mydata->val[0]);
        mydata->msg.data[index++] = half1 & 0xFF;
        mydata->msg.data[index++] = (half1 >> 8) & 0xFF;
        mydata->msg.data[index++] = (half1 >> 16) & 0xFF;
        int32_t const half2 = float_to_24bits(mydata->val[1]);
        mydata->msg.data[index++] = half2 & 0xFF;
        mydata->msg.data[index++] = (half2 >> 8) & 0xFF;
        mydata->msg.data[index++] = (half2 >> 16) & 0xFF;
#elif defined(ENABLE_TRIPLE_DIFF)
        int16_t const half1 = float_to_half(mydata->val[0]);
        mydata->msg.data[index++] = half1 & 0xFF;
        mydata->msg.data[index++] = half1 >> 8;
        int16_t const half2 = float_to_half(mydata->val[1]);
        mydata->msg.data[index++] = half2 & 0xFF;
        mydata->msg.data[index++] = half2 >> 8;
        int16_t const half3 = float_to_half(mydata->val[2]);
        mydata->msg.data[index++] = half3 & 0xFF;
        mydata->msg.data[index++] = half3 >> 8;

#elif defined(ENABLE_HEXA_DIFF)
        int8_t const half1 = float_to_i8bits(mydata->val[0]);
        mydata->msg.data[index++] = half1;
        int8_t const half2 = float_to_i8bits(mydata->val[1]);
        mydata->msg.data[index++] = half2;
        int8_t const half3 = float_to_i8bits(mydata->val[2]);
        mydata->msg.data[index++] = half3;
        int8_t const half4 = float_to_i8bits(mydata->val[3]);
        mydata->msg.data[index++] = half4;
        int8_t const half5 = float_to_i8bits(mydata->val[4]);
        mydata->msg.data[index++] = half5;
        int8_t const half6 = float_to_i8bits(mydata->val[5]);
        mydata->msg.data[index++] = half6;
#endif

#ifdef ENABLE_HANDSHAKES
    } else if(mydata->data_type == DATA_HANDSHAKE) {
        handshake_message_data_t* hmsg = (handshake_message_data_t*) &mydata->msg.data;
        // Compute number of peers to include in the message
        uint8_t nb_peers = 3;
        if(mydata->nb_neighbors < 3)
            nb_peers = mydata->nb_neighbors;
        hmsg->nb_peers = nb_peers;
        // Add peers in the message
        for(uint8_t i = 0; i < nb_peers; i++) {
            if(mydata->current_peer_index >= mydata->nb_neighbors)
                mydata->current_peer_index = 0;
            hmsg->peers[i] = mydata->neighbors[mydata->current_peer_index].ID;
            mydata->current_peer_index++;
        }
#ifdef SIMULATOR
//        if(kilo_uid == 1) {
//            printf("SETUP HANDSHAKE! nb_peers=%d nb_neighbors=%d current_peer_index=%d peer0=%d peer1=%d peer2=%d\n", nb_peers, mydata->nb_neighbors, mydata->current_peer_index, hmsg->peers[0], hmsg->peers[1], hmsg->peers[2]);
//        }
#endif
#endif

    } else {
        int32_t casted_val = to_uint(mydata->val[0]);
        mydata->msg.data[index++] = ((int32_t)casted_val) & 0xFF;
        mydata->msg.data[index++] = (((int32_t)casted_val)>>8)&0xFF;
        mydata->msg.data[index++] = (((int32_t)casted_val)>>16)&0xFF;
        mydata->msg.data[index++] = (((int32_t)casted_val)>>24)&0xFF;
    }

    mydata->msg.crc = message_crc(&mydata->msg);
}



/* Process a received message at the front of the ring buffer.
 * Go through the list of neighbors. If the message is from a bot
 * already in the list, update the information, otherwise
 * add a new entry in the list
 */
void process_message() {
    uint8_t index = 0;
    uint8_t i;
    uint16_t ID;

#ifdef SIMULATOR
    uint16_t d = estimate_distance_simulator(&RB_front().dist);
#else
    uint8_t d = estimate_distance(&RB_front().dist);
#endif
    if (d > CUTOFF)
        return;

    uint8_t *data = RB_front().msg.data;
    ID = data[index] | (data[index+1] << 8);
    index += 2;

    // Check if the message is not from the focal robot
    if(ID == kilo_uid)
        return;

    //data_type_t data_type = data[index++];
    data_type_t data_type = RB_front().msg.type;

#ifdef ENABLE_HANDSHAKES
    if(data_type != DATA_HANDSHAKE) {
        // Check if neighbor is known
        if(!is_neighbor_known(ID)) {
            // Unknown neighbor. Ignore message
            return;
        }
    }
#endif

    // search the neighbor list by ID
    for(i = 0; i < mydata->nb_neighbors; i++)
        if(mydata->neighbors[i].ID == ID) // found it
            break;

    if(i == mydata->nb_neighbors)   // this neighbor is not in list
        if(mydata->nb_neighbors < MAXN-1) // if we have too many neighbors, we overwrite the last entry
            mydata->nb_neighbors++;          // sloppy but better than overflow

    // i now points to where this message should be stored
    mydata->neighbors[i].ID = ID;
    mydata->neighbors[i].timestamp = kilo_ticks;
    mydata->neighbors[i].dist = d;
    mydata->neighbors[i].data_type = data_type;


    if(data_type == DATA_S) {
#if defined(ENABLE_SINGLE_DIFF)
        int32_t const orig_val = (int32_t)data[index] | (int32_t)data[index+1] << 8 | (int32_t)data[index+2] << 16 | (int32_t)data[index+3] << 24;
        float const casted_val = to_float(orig_val);
        mydata->neighbors[i].val[0] = casted_val;
        index += 4;
#elif defined(ENABLE_DOUBLE_DIFF)
        int32_t const orig_val1 = data[index] | (data[index+1] << 8) | ((int32_t)data[index+2] << 16);
        //float const casted_val1 = half_to_float(orig_val1);
        float const casted_val1 = i24bits_to_float(orig_val1);
        mydata->neighbors[i].val[0] = casted_val1;
        index += 3;
        int32_t const orig_val2 = data[index] | (data[index+1] << 8) | ((int32_t)data[index+2] << 16);
        //float const casted_val2 = half_to_float(orig_val2);
        float const casted_val2 = i24bits_to_float(orig_val2);
        mydata->neighbors[i].val[1] = casted_val2;
        index += 3;
#elif defined(ENABLE_TRIPLE_DIFF)
        int16_t const orig_val1 = data[index] | (data[index+1] << 8);
        float const casted_val1 = half_to_float(orig_val1);
        mydata->neighbors[i].val[0] = casted_val1;
        index += 2;
        int16_t const orig_val2 = data[index] | (data[index+1] << 8);
        float const casted_val2 = half_to_float(orig_val2);
        mydata->neighbors[i].val[1] = casted_val2;
        index += 2;
        int16_t const orig_val3 = data[index] | (data[index+1] << 8);
        float const casted_val3 = half_to_float(orig_val3);
        mydata->neighbors[i].val[2] = casted_val3;
        index += 2;

#elif defined(ENABLE_HEXA_DIFF)
        mydata->neighbors[i].val[0] = i8bits_to_float(data[index++]);
        mydata->neighbors[i].val[1] = i8bits_to_float(data[index++]);
        mydata->neighbors[i].val[2] = i8bits_to_float(data[index++]);
        mydata->neighbors[i].val[3] = i8bits_to_float(data[index++]);
        mydata->neighbors[i].val[4] = i8bits_to_float(data[index++]);
        mydata->neighbors[i].val[5] = i8bits_to_float(data[index++]);
#endif

#ifdef ENABLE_HANDSHAKES
    } else if(data_type == DATA_HANDSHAKE) {
        handshake_message_data_t const* hmsg = (handshake_message_data_t*) data;
        bool const am_i_a_peer =
               (hmsg->nb_peers > 0 && kilo_uid == hmsg->peers[0])
            || (hmsg->nb_peers > 1 && kilo_uid == hmsg->peers[1])
            || (hmsg->nb_peers > 2 && kilo_uid == hmsg->peers[2]);
        if(am_i_a_peer) {
            // Check if the neighbor is not already known
            if(!is_neighbor_known(ID) && mydata->nb_known_neighbors < MAXN-1) {
                // If neighbor is unknown, add it to the list of known neighbors
                mydata->known_neighbors_uid[mydata->nb_known_neighbors] = ID;
                mydata->nb_known_neighbors++;          // sloppy but better than overflow
            }
        }
#ifdef SIMULATOR
//        if(kilo_uid == 1) {
//            printf("PROCESS HANDSHAKE! am_i_a_peer=%d ID=%d nb_known_neighbors=%d\n", am_i_a_peer, ID, mydata->nb_known_neighbors);
//        }
#endif
#endif

    } else {
        int32_t const orig_val = (int32_t)data[index] | (int32_t)data[index+1] << 8 | (int32_t)data[index+2] << 16 | (int32_t)data[index+3] << 24;
        float const casted_val = to_float(orig_val);
        mydata->neighbors[i].val[0] = casted_val;
        index += 4;
    }
}





void setup_diff(diffusion_session_t* diff) {
    diff->t = 0;

    for(uint8_t i = 0; i < NUMBER_DIFF; i++) {
        diff->s[i] = 0;
        diff->s0[i] = 0;
#if defined(ENABLE_INIT_REMOVE_SUM_S)
        diff->sum_s0[i] = 0;
#endif
        diff->lambda_[i] = 0;
        diff->sum_t[i] = 0;
        diff->sum_t2[i] = 0;
        diff->sum_logs[i] = 0;
        diff->sum_tlogs[i] = 0;
        diff->ls_nb_points[i] = 0;
    }

    diff->lambda = 0;
    diff->sum_lambda = 0.f;
    diff->avg_lambda = 0;
    diff->current_diffusion_it = 0;

    diff->diffusion_valid = false;
    diff->current_diffusion_it = 0;
    diff->current_avg_it = 0;
    diff->time_last_diff_it = 0;
}




void setup() {

    // Initialization of the random generator
    while(get_voltage() == -1);
    rand_seed(rand_hard()+kilo_uid);

    // Init kiloticks_iteration
    kiloticks_iteration = kiloticks_randow_walk + kiloticks_diffusion + kiloticks_collective_avg_lambda + kiloticks_end_it_waiting_time + kiloticks_start_it_waiting_time
#ifdef ENABLE_AVG_AVG_LAMBDA
        + kiloticks_collective_avg_avg_lambda
#endif
#ifdef ENABLE_PRE_DIFFUSION
        + kiloticks_diffusion
#endif
#ifdef ENABLE_HANDSHAKES
        + kiloticks_handshake
#endif
        ;

    clearNeighbors();
#ifdef ENABLE_HANDSHAKES
    clearKnownNeighbors();
#endif

    // Init local variables
    for(uint8_t i = 0; i < NUMBER_DIFF; i++) {
        mydata->val[i] = 0;
    }
    mydata->data_type = DATA_NULL;
    mydata->time_last_it = 0;
    mydata->current_it = 0;
    mydata->kiloticks_start_it = 0;
    mydata->current_behavior = RANDOM_WALK;
    setup_diff(&mydata->diff1);
    mydata->curr_diff = &mydata->diff1;

#ifdef SIMULATOR
//    kilo_uid = rand_soft()%100;
#endif

#ifdef ENABLE_DISPERSION
    setup_dispersion();
#endif

    setup_message();
    set_color(RGB(3,3,3));
}


void behav_random_walk() {
    if(kilo_ticks % kiloticks_random_walk_choice == 0) {
        int r = rand() % 3;
        if(r==0) {
            set_motion(FORWARD);
        } else if(r==1) {
            set_motion(LEFT);
        } else if(r==2) {
            set_motion(RIGHT);
        }
    }
}


void init_diffusion(diffusion_session_t* diff, ext_t* s, diffusion_type_t type) {
    mydata->data_type = DATA_NULL;
    set_motion(STOP);
    clearNeighbors();
#ifdef SIMULATOR
    if(rnd_noise_dist_each_it > 0) {
        diff->noise_dist = (rand() % (2*rnd_noise_dist_each_it)) - rnd_noise_dist_each_it;
    }
#endif

    // Set 'diff' as current diffusion session
    mydata->curr_diff = diff;

    // Initialize diffusion information
    diff->type = type;
    diff->t = 0;

    for(uint8_t i = 0; i < NUMBER_DIFF; i++) {
        diff->s[i] = s[i];
        diff->s0[i] = diff->s[i];
        diff->lambda_[i] = 0.f;

        diff->sum_t[i] = 0.f;
        diff->sum_t2[i] = 0.f;
        diff->sum_logs[i] = 0.f;
        diff->sum_tlogs[i] = 0.f;
        diff->ls_nb_points[i] = 0;

        for(uint8_t j = 0; j < DIFFUSION_WINDOW_SIZE; j++) {
            diff->hist_logs[i][j] = 1000;
            diff->hist_t[i][j] = -1;
        }
        diff->best_mse[i] = 1000;
#ifdef SIMULATOR
        diff->last_mse[i] = -1.f;
#endif

        mydata->val[i] = diff->s[i];
    }
    diff->lambda = 0;
    diff->current_diffusion_it = 0;
    diff->time_last_diff_it = kilo_ticks;
    diff->diffusion_valid = true;

    // Broadcast s to neighboring agents
    mydata->data_type = DATA_S;

#ifdef ENABLE_COLOR_FROM_S0
    set_color_from_s(diff->s0[0]);
#elif defined(ENABLE_COLOR_FROM_S)
    set_color_from_s(diff->s0[0]);
#elif defined(ENABLE_COLOR_FROM_SIGNS)
    set_color_from_signs(diff->s0[0]);
#elif defined(ENABLE_COLOR_FROM_AVGLAMBDA)
//#ifdef ENABLE_PRE_DIFFUSION
//    set_color_from_lambda(mydata->diff1.avg_lambda);
//#else
    set_color_from_lambda(mydata->curr_diff->avg_lambda);
//#endif
#elif defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA) || defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA_ALL_DIFF)
    set_color_from_signs(diff->s0[0]);
#elif defined(ENABLE_COLOR_NB_NEIGHBOURS)
    set_color_from_nb_neighbours();
#elif defined(ENABLE_COLOR_FROM_SIGNDELTAX)
    set_color_from_signs(diff->s0[0]);
#endif

#ifdef SIMULATOR
    if(kilo_uid == 1) {
        printf("BEGIN DIFFUSION ! it=%d\n", mydata->current_it);
    }

    //printf("@@@@@@@@@@@@ NEW DIFFUSION ROUND @@@@@@@@@@@@\n");
#endif
}


void compute_next_s() {
#if defined(ENABLE_SINGLE_DIFF)
    ext_t new_s[1] = {0};
#elif defined(ENABLE_DOUBLE_DIFF)
    ext_t new_s[2] = {0, 0};
#elif defined(ENABLE_TRIPLE_DIFF)
    ext_t new_s[3] = {0, 0, 0};
#elif defined(ENABLE_HEXA_DIFF)
    ext_t new_s[6] = {0, 0, 0, 0, 0, 0};
#endif

#ifdef SIMULATOR
    uint32_t comms_radius = simparams->commsRadius;
    if(rnd_comms_radius > 0) {
        comms_radius = (rand() % (simparams->commsRadius - rnd_comms_radius +1)) + rnd_comms_radius;
    }
#else
    uint32_t const comms_radius = REALROBOTS_CUTOFF;
#endif
    uint16_t dist;

    for(uint8_t i = 0; i < mydata->nb_neighbors; i++) {
#ifdef SIMULATOR
        dist = mydata->neighbors[i].dist;
        if(rnd_noise_dist > 0) {
            if(rand() % 2) {
                dist -= rnd_noise_dist;
            } else {
                dist += rnd_noise_dist;
            }
            if(dist < 0)
                dist = 0;
        } else if(rnd_noise_dist_each_it > 0) {
            dist += mydata->curr_diff->noise_dist;
        }
#else
        dist = mydata->neighbors[i].dist;
#endif

        if( mydata->neighbors[i].data_type == DATA_S && dist < comms_radius ) {
            for(uint8_t j = 0; j < NUMBER_DIFF; j++) {
                ext_t const val = mydata->neighbors[i].val[j];
                // Step Kernel
                if(is_number_valid(val)) {
                    new_s[j] += (ext_t)(mydata->neighbors[i].val[j] - mydata->curr_diff->s[j]);
                }
            }
        }
    }

#if defined(ENABLE_COLOR_NB_NEIGHBOURS)
    set_color_from_nb_neighbours();
    //printf("next_s: new_s %f\n", new_s);
#endif
    clearNeighbors();
    mydata->curr_diff->s[0] += new_s[0] / (ext_t)inv_tau;
#if defined(ENABLE_DOUBLE_DIFF)
    mydata->curr_diff->s[1] += new_s[1] / (ext_t)inv_tau;
#elif defined(ENABLE_TRIPLE_DIFF)
    mydata->curr_diff->s[1] += new_s[1] / (ext_t)inv_tau;
    mydata->curr_diff->s[2] += new_s[2] / (ext_t)inv_tau;
#elif defined(ENABLE_HEXA_DIFF)
    mydata->curr_diff->s[1] += new_s[1] / (ext_t)inv_tau;
    mydata->curr_diff->s[2] += new_s[2] / (ext_t)inv_tau;
    mydata->curr_diff->s[3] += new_s[3] / (ext_t)inv_tau;
    mydata->curr_diff->s[4] += new_s[4] / (ext_t)inv_tau;
    mydata->curr_diff->s[5] += new_s[5] / (ext_t)inv_tau;
#endif

    // Check that s is not out of bounds
    for(uint8_t j = 0; j < NUMBER_DIFF; j++) {
        if(fabs(mydata->curr_diff->s[j]) > 2* initial_s_max_val) {
            mydata->curr_diff->diffusion_valid = false;
            mydata->curr_diff->s[j] = 0.f/0.f; // NaN
        }
    }
}





ext_t compute_MSE(ext_t lambda, ext_t v, ext_t* hist_logs, ext_t* hist_t) {
    ext_t mse = 0;
    ext_t nb_points = 0;
    for(uint8_t i = 0; i < DIFFUSION_WINDOW_SIZE; i++) {
        ext_t const logs = hist_logs[i];
        ext_t const t = hist_t[i];
        if(logs >= 0 || t < 0)
            continue;
        ext_t const err = (-lambda * t + log(v)) - (logs);
        //ext_t const err = (-lambda * t + v) - (logs);
        if(is_number_valid(err)) {
            mse += err * err;
            nb_points += 1;
        }
    }
    if(nb_points > 0)
        return mse/nb_points;
    else
        return 2000;
}


void compute_lambda_v_leastsquaresMSE() {
    diffusion_session_t *const diff = mydata->curr_diff;
    ext_t const t = (ext_t)diff->current_diffusion_it / inv_tau;
    diff->t = t;

    ext_t nb_valid_lambda = 0.f;
    ext_t sum_all_lambda = 0.f;

    if(diff->diffusion_valid && diff->current_diffusion_it >= kiloticks_diffusion_burnin / kiloticks_diffusion_it) {

        for(uint8_t i = 0; i < NUMBER_DIFF; i++) {
            ext_t mse = 1000;
            ext_t const logs = log(fabs(diff->s[i]));
            if(is_number_valid(logs) && fabs(logs) > 0.0) {
                diff->sum_t[i] += t;
                diff->sum_t2[i] += t * t;
                diff->sum_logs[i] += logs;
                diff->sum_tlogs[i] += t * logs;
                diff->ls_nb_points[i] += 1.f;
                diff->hist_logs[i][(uint8_t)(diff->ls_nb_points[i]) % DIFFUSION_WINDOW_SIZE] = logs;
                diff->hist_t[i][(uint8_t)(diff->ls_nb_points[i]) % DIFFUSION_WINDOW_SIZE] = t;

                if(diff->ls_nb_points[i] > 3) {
                    ext_t const _lambda = -(diff->ls_nb_points[i] * diff->sum_tlogs[i] - diff->sum_t[i] * diff->sum_logs[i]) / (diff->ls_nb_points[i] * diff->sum_t2[i] - diff->sum_t[i] * diff->sum_t[i]);
                    ext_t const _v = exp( (diff->sum_logs[i] - _lambda * diff->sum_t[i]) / diff->ls_nb_points[i] );

                    if(!is_number_valid(_lambda) || !is_number_valid(_v)) {
                        // ...
                        //diff->diffusion_valid = false;
                    } else {
                        if(diff->ls_nb_points[i] >= DIFFUSION_WINDOW_SIZE) {
                            mse = compute_MSE(_lambda, _v, diff->hist_logs[i], diff->hist_t[i]);
                            if(mse < diff->best_mse[i]) {
                                diff->best_mse[i] = mse;
                                diff->lambda_[i] = _lambda;
                                //diff->v[i] = _v;
                            }
                        } else {
                            diff->lambda_[i] = _lambda;
                            //diff->v[i] = _v;
                        }

                        if(is_number_valid(diff->lambda_[i]) && diff->best_mse[i] < 100.f) {
                            sum_all_lambda += diff->lambda_[i];
                            nb_valid_lambda++;
                        }
                    }

                }
            }
        }

        ext_t const _lambda = sum_all_lambda / nb_valid_lambda;
        if(!is_number_valid(_lambda) || nb_valid_lambda == 0) {
//            diff->diffusion_valid = false;
        } else {
            diff->lambda = _lambda;
        }
    }


#ifdef SIMULATOR
//    if(kilo_uid == 1) {
//        printf("DEBUG diff: valid=%d t=%f s=%f logs=%f orig_t=%f orig_logx=%f lambda=%f v=%f cv=%f \n", diff->diffusion_valid, (float)t, (float)diff->s, (float)logs, (float)diff->diffusion_orig_t, (float)diff->diffusion_orig_logx, (float)diff->lambda, (float)diff->v, (float)diff->cv);
//    }
#endif
}


void behav_diffusion() {
    diffusion_session_t *const diff = mydata->curr_diff;
    set_motion(STOP);

    if(kilo_ticks - diff->time_last_diff_it >= kiloticks_diffusion_it) {
        compute_next_s();
        if(diff->type != PRE_DIFFUSION_TYPE) {
            compute_lambda_v_leastsquaresMSE();
        }

        // Update broadcasting information
        for(uint8_t i = 0; i < NUMBER_DIFF; i++) {
            mydata->val[i] = diff->s[i];
        }
        mydata->data_type = DATA_S;

        diff->current_diffusion_it++;
        diff->time_last_diff_it = kilo_ticks;

#if defined(ENABLE_COLOR_FROM_LAMBDA)
        set_color_from_lambda(diff->lambda);
#elif defined(ENABLE_COLOR_FROM_S)
        set_color_from_s(diff->s[0]);
#elif defined(ENABLE_COLOR_FROM_SIGNS)
        set_color_from_signs(diff->s[0]);
#elif defined(ENABLE_COLOR_FROM_AVGLAMBDA)
//#ifdef ENABLE_PRE_DIFFUSION
//        set_color_from_lambda(mydata->diff1.avg_lambda);
//#else
        set_color_from_lambda(mydata->curr_diff->avg_lambda);
//#endif
#elif defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA)
        set_color_from_signs(diff->s[0]);
#elif defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA_ALL_DIFF)
#if defined(ENABLE_SINGLE_DIFF)
        set_color_from_signs(diff->s[0]);
#elif defined(ENABLE_DOUBLE_DIFF)
        if((diff->current_diffusion_it / 7) % 2 == 0) {
            set_color_from_signs(diff->s[0]);
        } else {
            set_color_from_signs(diff->s[1]);
        }
#elif defined(ENABLE_TRIPLE_DIFF)
        if((diff->current_diffusion_it / 7) % 3 == 0) {
            set_color_from_signs(diff->s[0]);
        } else if((diff->current_diffusion_it / 7) % 3 == 1) {
            set_color_from_signs(diff->s[1]);
        } else {
            set_color_from_signs(diff->s[2]);
        }
#elif defined(ENABLE_HEXA_DIFF)
        if((diff->current_diffusion_it / 3) % 6 == 0) {
            set_color_from_signs(diff->s[0]);
        } else if((diff->current_diffusion_it / 3) % 6 == 1) {
            set_color_from_signs(diff->s[1]);
        } else if((diff->current_diffusion_it / 3) % 6 == 2) {
            set_color_from_signs(diff->s[2]);
        } else if((diff->current_diffusion_it / 3) % 6 == 3) {
            set_color_from_signs(diff->s[3]);
        } else if((diff->current_diffusion_it / 3) % 6 == 4) {
            set_color_from_signs(diff->s[4]);
        } else {
            set_color_from_signs(diff->s[5]);
        }
#endif

#elif defined(ENABLE_COLOR_NB_NEIGHBOURS)
        set_color_from_nb_neighbours();
#endif

    }
    setup_message();
}



void end_diffusion() {
    diffusion_session_t *const diff = mydata->curr_diff;

    // Check if diffusion converged close to 0.
    //if(fabs(diff->s) - 0. > diffusion_convergence_threshold) {
    //if(fabs(diff->s) - 0. > diffusion_convergence_threshold || diff->ls_nb_points < diffusion_min_nb_points) {
    if( (diffusion_convergence_threshold > 0. && ABS(diff->s[0]) - 0. > diffusion_convergence_threshold)
            || diff->ls_nb_points[0] < diffusion_min_nb_points
#if defined(ENABLE_DOUBLE_DIFF)
            || diff->ls_nb_points[1] < diffusion_min_nb_points
#elif defined(ENABLE_TRIPLE_DIFF)
            || diff->ls_nb_points[1] < diffusion_min_nb_points
            || diff->ls_nb_points[2] < diffusion_min_nb_points
#elif defined(ENABLE_HEXA_DIFF)
            || diff->ls_nb_points[1] < diffusion_min_nb_points
            || diff->ls_nb_points[2] < diffusion_min_nb_points
            || diff->ls_nb_points[3] < diffusion_min_nb_points
            || diff->ls_nb_points[4] < diffusion_min_nb_points
            || diff->ls_nb_points[5] < diffusion_min_nb_points
#endif
            || !is_number_valid(diff->lambda)
            ) {
        diff->diffusion_valid = false;
    }
    if(!is_number_valid(diff->lambda)) {
        diff->lambda = 0;
        diff->diffusion_valid = false;
    }

    if(!diff->diffusion_valid) {
        diff->lambda = 0.0/0.0; // NaN
    }

#ifdef SIMULATOR
    for(uint8_t i = 0; i < NUMBER_DIFF; i++) {
        if(diff->best_mse[i] < 100.f) {
            diff->last_mse[i] = diff->best_mse[i];
        }
    }

    if(kilo_uid == 1) {
        printf("END DIFFUSION ! ticks=%d it=%d uid=%d diffit=%d lambda=%f s[0]=%f nb_points1=%f\n", kilo_ticks - diff->time_last_diff_it, mydata->current_it, kilo_uid, diff->current_diffusion_it, (float)diff->lambda, (float)diff->s[0], diff->ls_nb_points[0]);
    }
#endif
}


void init_coll_avg_lambda() {
    mydata->data_type = DATA_NULL;
    set_motion(STOP);
    clearNeighbors();
    mydata->time_last_coll_avg_lambda_it = kilo_ticks;
    mydata->current_coll_avg_step = 0;

    diffusion_session_t *const diff = mydata->curr_diff;
    if(diff->diffusion_valid) {
        mydata->val[0] = diff->lambda;
        mydata->data_type = DATA_LAMBDA;
    } else {
        mydata->data_type = DATA_NULL;
    }
}

void behav_coll_avg_lambda() {
    set_motion(STOP);
    if(kilo_ticks - mydata->time_last_coll_avg_lambda_it >= kiloticks_collective_avg_lambda_it) {
        // Compute lambda through collective averaging
        uint8_t nb_neighbors = mydata->nb_neighbors;
        uint8_t used_neighbors = nb_neighbors;
        ext_t tmp = 0;
        if(mydata->curr_diff->diffusion_valid) {
            tmp += mydata->curr_diff->lambda;
            ++used_neighbors;
        }
        for(uint8_t i = 0; i < nb_neighbors; i++) {
            if(mydata->neighbors[i].data_type == DATA_LAMBDA) {
                tmp += mydata->neighbors[i].val[0];
            } else {
                --used_neighbors;
            }
        }

        clearNeighbors();
        if(used_neighbors > 0) {
            tmp /= (ext_t)(used_neighbors);
            if(is_number_valid(tmp)) {
                mydata->curr_diff->lambda = tmp;
                mydata->val[0] = mydata->curr_diff->lambda;
                mydata->data_type = DATA_LAMBDA;
            }
        }

        mydata->current_coll_avg_step++;
        mydata->time_last_coll_avg_lambda_it = kilo_ticks;

#ifdef SIMULATOR
//        if(kilo_uid == 1) {
//            printf("DEBUG avg_lambda: lambda=%f tmp=%f\n", (float)mydata->curr_diff->lambda, (float)tmp);
//        }
#endif
    }
    setup_message();
}

void end_coll_avg_lambda() {
#ifndef ENABLE_AVG_AVG_LAMBDA
    diffusion_session_t *const diff = mydata->curr_diff;
    if(is_number_valid(diff->lambda)) {
        diff->sum_lambda += diff->lambda * 1;
        ++diff->current_avg_it;
        diff->avg_lambda = (ext_t) diff->sum_lambda / (diff->current_avg_it);
#if defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA) || defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA_ALL_DIFF)
        set_color_from_lambda(diff->avg_lambda);
#endif
    }

#ifdef SIMULATOR
    if(kilo_uid == 1) {
        printf("DEBUG end_coll_avg_lambda: sum_lambda=%f avg_lambda=%f\n", (float)diff->sum_lambda, (float)diff->avg_lambda);
    }
#endif
#endif
}


#ifdef ENABLE_AVG_AVG_LAMBDA
void init_coll_avg_avg_lambda() {
    mydata->data_type = DATA_NULL;
    set_motion(STOP);
    clearNeighbors();
    mydata->current_coll_avg_step = 0;

    diffusion_session_t *const diff = mydata->curr_diff;
    mydata->time_last_coll_avg_avg_lambda_it = kilo_ticks;

    if(is_number_valid(diff->lambda)) {
        diff->sum_lambda += diff->lambda * 1;
        ++diff->current_avg_it;
        diff->avg_lambda = (ext_t) (diff->sum_lambda / (mydata->current_it + 1));
    }

    if(is_number_valid(diff->avg_lambda)) {
        mydata->val[0] = diff->avg_lambda;
        mydata->data_type = DATA_AVG_LAMBDA;
    } else {
        mydata->data_type = DATA_NULL;
    }
}

void behav_coll_avg_avg_lambda() {
    set_motion(STOP);
    diffusion_session_t *const diff = mydata->curr_diff;
    if(kilo_ticks - mydata->time_last_coll_avg_avg_lambda_it >= kiloticks_collective_avg_avg_lambda_it) {

        uint8_t nb_neighbors = mydata->nb_neighbors;
        uint8_t used_neighbors = nb_neighbors;
        float tmp = 0;
        if(is_number_valid(diff->avg_lambda)) {
            tmp += diff->avg_lambda;
            ++used_neighbors;
        }
        for(uint8_t i = 0; i < nb_neighbors; i++) {
            if(mydata->neighbors[i].data_type == DATA_AVG_LAMBDA) {
                tmp += mydata->neighbors[i].val[0];
            } else {
                --used_neighbors;
            }
        }

        clearNeighbors();
        if(used_neighbors > 0) {
            tmp /= (float)(used_neighbors);
            diff->avg_lambda = (ext_t)tmp;
            if(is_number_valid(diff->avg_lambda)) {
                mydata->val[0] = diff->avg_lambda;
                mydata->data_type = DATA_AVG_LAMBDA;
            }
        }
        mydata->current_coll_avg_step++;
        mydata->time_last_coll_avg_avg_lambda_it = kilo_ticks;
    }
    setup_message();

#if defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA) || defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA_ALL_DIFF)
    set_color_from_lambda(diff->avg_lambda);
#elif defined(ENABLE_COLOR_FROM_AVGLAMBDA)
    set_color_from_lambda(mydata->diff1.avg_lambda);
#endif

#ifdef SIMULATOR
//    if(kilo_uid == 1) {
//        printf("DEBUG avg_avg_lambda: avg_lambda=%f\n", (float)diff->avg_lambda);
//    }
#endif
}
#endif




void end_iteration() {
#if !defined(ENABLE_AVG_AVG_LAMBDA)
    end_coll_avg_lambda();
#endif

#ifdef ENABLE_HANDSHAKES
    clearKnownNeighbors();
#endif

#ifdef ENABLE_INIT_REMOVE_SUM_S
    if(mydata->current_it == 0) {
        diffusion_session_t *const diff = mydata->curr_diff;
        diff->sum_s0[0] = diff->s[0];
        diff->sum_s0[1] = diff->s[1];
        diff->sum_s0[2] = diff->s[2];
        diff->sum_lambda = 0.f;
        diff->avg_lambda = 0;
        diff->current_avg_it = 0;
#ifdef SIMULATOR
        if(kilo_uid == 1) {
            printf("SUM(S(0)):%f  x:%f  lambda:%f\n", diff->sum_s0, diff->s[0], diff->lambda);
        }
#endif
    }
#endif

#ifdef ENABLE_COLOR_FROM_AVGLAMBDA
    set_color_from_lambda(mydata->diff1.avg_lambda);
#elif defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA) || defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA_ALL_DIFF)
    set_color_from_lambda(mydata->curr_diff->avg_lambda);
#else
    set_color(RGB(3,3,3));
#endif
    mydata->current_it++;

#ifdef SIMULATOR
    if(kilo_uid == 1) {
        printf("END ITERATION: ticks=%d avg_lambda:%f\n", kilo_ticks - mydata->time_last_it, (float)mydata->curr_diff->avg_lambda);
    }
#endif
    mydata->time_last_it = kilo_ticks;
}


void iteration() {
    uint32_t index = 0;

    uint32_t const ticks_after_init_phase = kilo_ticks - kiloticks_initial_random_walk;

    // Determine current behavior
    uint32_t const it_ticks = ticks_after_init_phase - mydata->kiloticks_start_it;

    if(kilo_ticks < kiloticks_initial_random_walk) {
#ifdef ENABLE_DISPERSION
        if(mydata->current_behavior != RANDOM_WALK) {
            start_dispersion();
        }
#endif
        mydata->current_behavior = RANDOM_WALK;
        mydata->time_last_it = kilo_ticks;

    } else if(kiloticks_start_it_waiting_time != 0 && it_ticks < (index += kiloticks_start_it_waiting_time)) {
        if(mydata->current_behavior != WAITING_TIME_2) {
            set_color(RGB(3,3,3));
        }
        mydata->current_behavior = WAITING_TIME_2;

    } else if(it_ticks < (index += kiloticks_randow_walk)) {
        if(mydata->current_behavior != RANDOM_WALK && mydata->current_behavior != WAITING_TIME) {
#ifdef SIMULATOR
//            if(kilo_uid == 1) {
//                printf("DEBUG END IT: it_ticks=%d/%d\n", it_ticks, kiloticks_iteration);
//            }
#endif
        }
#ifdef ENABLE_DISPERSION
        if(mydata->current_behavior != RANDOM_WALK) {
            start_dispersion();
        }
#endif
        if(it_ticks == 0) {
#ifdef SIMULATOR
            if(kilo_uid == 1) {
                printf("BEGIN ITERATION ! it=%d\n", mydata->current_it);
            }
#endif
        }
        mydata->current_behavior = RANDOM_WALK;

#ifdef ENABLE_HANDSHAKES
    } else if(it_ticks < (index += kiloticks_handshake)) {
        if(mydata->current_behavior != HANDSHAKE) {
            set_motion(STOP);
            clearNeighbors();
            clearKnownNeighbors();
        }
        mydata->current_behavior = HANDSHAKE;
#endif

#ifdef ENABLE_PRE_DIFFUSION
    } else if(it_ticks < (index += kiloticks_diffusion)) {
        if(mydata->current_behavior != PRE_DIFFUSION) {
            //ext_t s = (kilo_uid % 2 == 0) ? -initial_s_max_val : initial_s_max_val;
            ext_t s[NUMBER_DIFF];
            for(uint8_t i = 0; i < NUMBER_DIFF; i++) {
                s[i] = ((uint32_t)(rand_soft()+kilo_uid) % 2 == 0) ? -initial_s_max_val : initial_s_max_val;
            }
            init_diffusion(&mydata->diff1, s, PRE_DIFFUSION_TYPE);
        }
        mydata->current_behavior = PRE_DIFFUSION;
#endif

    } else if(it_ticks < (index += kiloticks_diffusion)) {
        if(mydata->current_behavior != DIFFUSION) {
            ext_t s[NUMBER_DIFF];
            for(uint8_t i = 0; i < NUMBER_DIFF; i++) {
            //ext_t s = ((uint32_t)(rand_soft()+kilo_uid) % 2 == 0) ? -initial_s_max_val : initial_s_max_val;
                s[i] = (kilo_uid % 2 == 0) ? -initial_s_max_val : initial_s_max_val;
#ifdef ENABLE_INIT_REMOVE_SUM_S
                s[i] -= mydata->diff1.sum_s0[i];
#elif defined(ENABLE_PRE_DIFFUSION)
                s[i] = mydata->diff1.s0[i] - mydata->diff1.s[i];
#endif
            }
            init_diffusion(&mydata->diff1, s, NORMAL_DIFFUSION_TYPE);
        }
        mydata->current_behavior = DIFFUSION;

#ifdef ENABLE_INIT_REMOVE_SUM_S
    } else if(mydata->current_it > 0 && it_ticks < (index += kiloticks_collective_avg_lambda)) {
#else
    } else if(it_ticks < (index += kiloticks_collective_avg_lambda)) {
#endif
        if(mydata->current_behavior != AVG_LAMBDA) {
            end_diffusion();
            init_coll_avg_lambda();
        }
        mydata->current_behavior = AVG_LAMBDA;

#ifdef ENABLE_AVG_AVG_LAMBDA
#ifdef ENABLE_INIT_REMOVE_SUM_S
    } else if(mydata->current_it > 0 && it_ticks < (index += kiloticks_collective_avg_avg_lambda)) {
#else
    } else if(it_ticks < (index += kiloticks_collective_avg_avg_lambda)) {
#endif
        if(mydata->current_behavior != AVG_AVG_LAMBDA) {
            end_coll_avg_lambda();
            init_coll_avg_avg_lambda();
        }
        mydata->current_behavior = AVG_AVG_LAMBDA;
#endif

    //} else if(kiloticks_end_it_waiting_time != 0 && it_ticks < (index += kiloticks_end_it_waiting_time)) {
    } else if(it_ticks < (index += kiloticks_end_it_waiting_time)) {
//        if(mydata->current_behavior != WAITING_TIME) {
//            // End of an iteration !
//            end_iteration();
//        }
        mydata->current_behavior = WAITING_TIME;

    } else {
        // End of an iteration !
        end_iteration();
        mydata->kiloticks_start_it = kilo_ticks - kiloticks_initial_random_walk;
    }

    if(mydata->current_behavior == RANDOM_WALK) {
#ifdef ENABLE_DISPERSION
        behav_dispersion();
#else
        behav_random_walk();
#endif
#ifdef ENABLE_HANDSHAKES
    } else if(mydata->current_behavior == HANDSHAKE) {
        mydata->data_type = DATA_HANDSHAKE;
        setup_message();
#endif
    } else if(mydata->current_behavior == DIFFUSION) {
        behav_diffusion();
#ifdef ENABLE_PRE_DIFFUSION
    } else if(mydata->current_behavior == PRE_DIFFUSION) {
        behav_diffusion();
#endif
    } else if(mydata->current_behavior == AVG_LAMBDA) {
        behav_coll_avg_lambda();
#ifdef ENABLE_AVG_AVG_LAMBDA
    } else if(mydata->current_behavior == AVG_AVG_LAMBDA) {
        behav_coll_avg_avg_lambda();
#endif
    }
}


void loop() {
    purgeNeighbors();

    // Process messages in the RX ring buffer
    while (!RB_empty()) {
        process_message();
        RB_popfront();
    }

    iteration();
}



int main(void) {
    // initialize hardware
    kilo_init();

    // initialize ring buffer
    RB_init();

    // register message callbacks
    kilo_message_rx = rxbuffer_push;
    kilo_message_tx = message_tx;

    // register your program
    kilo_start(setup, loop);

    SET_CALLBACK(global_setup, global_setup);
    SET_CALLBACK(botinfo, botinfo);
    SET_CALLBACK(reset, setup);
    SET_CALLBACK(json_state, json_state);
    SET_CALLBACK(obstacles, obstacles_walls);

    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
