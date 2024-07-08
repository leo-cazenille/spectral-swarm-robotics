
#ifndef LIMMSSWARM_H_
#define LIMMSSWARM_H_

#ifdef SIMULATOR
#define MAXN 1000 // 20
#else
#define MAXN 20 // 20
#endif
#define RB_SIZE 8   // Ring buffer size. Choose a power of two for faster code

#define REALROBOTS_CUTOFF 85 // 70 // 85 // 60 // 80 // 85
//#define CUTOFF 85 //neighbors further away are ignored. (mm)
#ifdef SIMULATOR
#define CUTOFF 10000 //neighbors further away are ignored. (mm)
#else
#define CUTOFF REALROBOTS_CUTOFF //neighbors further away are ignored. (mm)
#endif

#define ENABLE_AVG_AVG_LAMBDA
//#define ENABLE_INIT_REMOVE_SUM_S
#define ENABLE_PRE_DIFFUSION
#define ENABLE_HANDSHAKES
#define ENABLE_DISPERSION

//#define ENABLE_SINGLE_DIFF
//#define ENABLE_DOUBLE_DIFF
#define ENABLE_TRIPLE_DIFF
//#define ENABLE_HEXA_DIFF

#if !defined(ENABLE_COLOR_FROM_S0) && !defined(ENABLE_COLOR_FROM_S) && !defined(ENABLE_COLOR_FROM_SIGNS) && !defined(ENABLE_COLOR_FROM_LAMBDA) && !defined(ENABLE_COLOR_FROM_AVGLAMBDA) && !defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA) && !defined(ENABLE_COLOR_SIGNS_AND_AVGLAMBDA_ALL_DIFF) && !defined(ENABLE_COLOR_NB_NEIGHBOURS)
//#define ENABLE_COLOR_FROM_S0
//#define ENABLE_COLOR_FROM_S
//#define ENABLE_COLOR_FROM_SIGNS
//#define ENABLE_COLOR_FROM_LAMBDA
//#define ENABLE_COLOR_FROM_AVGLAMBDA
#define ENABLE_COLOR_SIGNS_AND_AVGLAMBDA
//#define ENABLE_COLOR_SIGNS_AND_AVGLAMBDA_ALL_DIFF
//#define ENABLE_COLOR_NB_NEIGHBOURS
#endif

//#define SPECIFY_CENTROIDS_IN_CONFIG


#define DIFFUSION_WINDOW_SIZE 20
//#define DIFFUSION_WINDOW_SIZE 10
//#define DIFFUSION_WINDOW_SIZE 3

#if defined(ENABLE_SINGLE_DIFF)
#define NUMBER_DIFF 1
#elif defined(ENABLE_DOUBLE_DIFF)
#define NUMBER_DIFF 2
#elif defined(ENABLE_TRIPLE_DIFF)
#define NUMBER_DIFF 3
#elif defined(ENABLE_HEXA_DIFF)
#define NUMBER_DIFF 6
#endif

typedef float base_t;
typedef float ext_t;
typedef float float_t;
#define ABS(x) fabs(x)

#ifdef SIMULATOR
extern float initial_s_max_val;
#else
extern float const initial_s_max_val;
#endif
void init_params();


// Behaviors
typedef enum {
    RANDOM_WALK,
#ifdef ENABLE_HANDSHAKES
    HANDSHAKE,
#endif
    PRE_DIFFUSION,
    DIFFUSION,
    AVG_LAMBDA,
#ifdef ENABLE_AVG_AVG_LAMBDA
    AVG_AVG_LAMBDA,
#endif
    WAITING_TIME,
    WAITING_TIME_2,
} behavior_t;


// declare variables

typedef enum {
    DATA_NULL = 5,  // Cf kilolib/message.h --> range btw 0x02 and 0x79 are available
#ifdef ENABLE_HANDSHAKES
    DATA_HANDSHAKE,
#endif
    DATA_S,
    DATA_LAMBDA,
    DATA_AVG_LAMBDA,
} data_type_t;

typedef struct {
    uint16_t ID;
#ifdef SIMULATOR
    uint16_t dist;
#else
    uint8_t dist;
#endif
    uint32_t timestamp;

    ext_t val[NUMBER_DIFF];
    data_type_t data_type;

    uint16_t coll_avg_step;
} neighbor_t;

typedef struct {
    message_t msg;
    distance_measurement_t dist;
} received_message_t;

#ifdef ENABLE_HANDSHAKES
#pragma pack(1)                                      // These two lines are needed to ensure 
typedef  struct  __attribute__((__packed__)) {       //   that all variable follow the same order as defined in the code
    uint16_t uid;  // uid of the sender
    uint16_t peers[3];
    uint8_t nb_peers;
    //uint8_t data[1]; ///< Rest of the message payload.  MUST be 9 - sizeof(uid) - sizeof(something) - sizeof(something_bigger)
} handshake_message_data_t;
#endif


typedef enum {
    NORMAL_DIFFUSION_TYPE = 0,
    PRE_DIFFUSION_TYPE,
} diffusion_type_t;


typedef struct {
    diffusion_type_t type;

    ext_t t;
    ext_t s[NUMBER_DIFF];
    ext_t s0[NUMBER_DIFF];

#if defined(ENABLE_INIT_REMOVE_SUM_S)
    ext_t sum_s0[NUMBER_DIFF];
#endif

#ifdef SIMULATOR
    uint16_t noise_dist;
#endif

    ext_t lambda;
    ext_t lambda_[NUMBER_DIFF];
    ext_t avg_lambda;
    ext_t sum_lambda;

    ext_t sum_t[NUMBER_DIFF];
    ext_t sum_t2[NUMBER_DIFF];
    ext_t sum_logs[NUMBER_DIFF];
    ext_t sum_tlogs[NUMBER_DIFF];
    ext_t ls_nb_points[NUMBER_DIFF];

    ext_t hist_logs[NUMBER_DIFF][DIFFUSION_WINDOW_SIZE];
    ext_t hist_t[NUMBER_DIFF][DIFFUSION_WINDOW_SIZE];
    ext_t best_mse[NUMBER_DIFF];
#ifdef SIMULATOR
    ext_t last_mse[NUMBER_DIFF];
#endif

    uint16_t current_avg_it;
    uint8_t diffusion_valid;

    uint16_t current_diffusion_it;
    uint32_t time_last_diff_it;
} diffusion_session_t;

typedef struct {
    neighbor_t neighbors[MAXN];
    uint8_t nb_neighbors;

#ifdef ENABLE_HANDSHAKES
    uint16_t known_neighbors_uid[MAXN];
    uint8_t nb_known_neighbors;
    uint8_t current_peer_index;
#endif

    message_t msg;

    received_message_t RXBuffer[RB_SIZE];
    uint8_t RXHead, RXTail;

    data_type_t data_type;

    ext_t val[NUMBER_DIFF];

    diffusion_session_t diff1;
    diffusion_session_t* curr_diff;

    uint32_t time_last_it;
    uint32_t time_last_coll_avg_lambda_it;
#ifdef ENABLE_AVG_AVG_LAMBDA
    uint32_t time_last_coll_avg_avg_lambda_it;
#endif

    uint32_t kiloticks_start_it;
    behavior_t current_behavior;
    uint16_t current_it;
    uint16_t current_coll_avg_step;

    // Dispersion
    uint16_t cycle_dispersion;
    uint16_t last_kiloticks_dispersion;
    uint16_t tumble_time;
    uint16_t run_time;
    uint8_t direction;
    float prob;
    uint8_t flag_dispersion;
    float d_min;
    float d_max;
    float frustration;


} USERDATA;

extern USERDATA *mydata;



#endif

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
