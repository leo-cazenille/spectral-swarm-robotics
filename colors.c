/* TODO
 */

#include <kilombo.h>
#include "colors.h"
#include <math.h>


#define NRAINBOWCOLORS (sizeof(colors) / sizeof((colors)[0]))

//// Double rainbow colors (all the way !)
//uint8_t colors[] = {
//    RGB(3,0,0),  //0 - red
//    RGB(3,3,0),  //3 - yellow
//    RGB(0,3,0),  //4 - green
//    RGB(0,3,3),  //4 - green
//    RGB(0,0,3),  //6 - blue
//
//    RGB(2,2,2),  //8 - grey
//    RGB(0,0,0),  //8 - off
//    RGB(2,1,0),  //8 - orange
//    RGB(1,1,1),  //8 - grey
//
//    RGB(2,0,0),  //0 - red
//    RGB(2,2,0),  //2 - yellow
//    RGB(0,2,0),  //4 - green
//    RGB(0,2,2),  //4 - green
//    RGB(0,0,2),  //6 - blue
//};

// Generated using the following Python commands:
//  a = np.linspace(0., 1.0, 100)
//  hsv = [np.around(matplotlib.colors.hsv_to_rgb((a, 1.0, 1.0)) * 3) for a in a]  
//  colors = set([tuple(c.astype(int).tolist()) for c in hsv])
uint8_t colors[] = {
    RGB(0, 0, 3),
    RGB(0, 1, 3),
    RGB(0, 2, 3),
    RGB(0, 3, 0),
    RGB(0, 3, 1),
    RGB(0, 3, 2),
    RGB(0, 3, 3),
    RGB(1, 0, 3),
    RGB(1, 3, 0),
    RGB(2, 0, 3),
    RGB(2, 3, 0),
    RGB(3, 0, 0),
    RGB(3, 0, 1),
    RGB(3, 0, 2),
    RGB(3, 0, 3),
    RGB(3, 1, 0),
    RGB(3, 2, 0),
    RGB(3, 3, 0),
};


#define NCLASSES (sizeof(class_centroids) / sizeof((class_centroids)[0]))

#ifdef SPECIFY_CENTROIDS_IN_CONFIG
// Config final_triplediff-MSE-HANDSHAKE-diff54250-burnin10000-msg03-ag25-fop85-it248-tau50-expe5000-50runs_expeIC_danda.yaml
float class_centroids[] = {
    0.0001,
    0.01,
    0.40343048418760297,
    0.6576639905929565,

    -1000.0,
    -1000.0,
    -1000.0,
    -1000.0,
    -1000.0,
    -1000.0,
};

// Cf dict 'colors_dict' in scripts/plots.py.
// To generate kilobots colors:
//   colors_k = {k: np.around(np.array(matplotlib.colors.to_rgb(v)) * 3.).astype(int) for k,v in plots.colors_dict.items()}
uint8_t class_colors[] = {
    RGB(0,3,0), // green (= error!)
    RGB(2,0,2), // purple - annulus
    RGB(2,0,2), // purple - annulus
    RGB(0,2,2), // cyan - disk

    RGB(3,1,0), // orange - square
    RGB(1,2,1), // dark green - arrow2
    RGB(3,0,0), // red - star
    RGB(2,1,1), // brown - 8
    RGB(1,1,1), // grey - 6
    RGB(2,2,0), // yellow - triangle
};


#else

// Config final_triplediff-MSE-HANDSHAKE-diff54250-burnin10000-msg03-ag25-fop85-it248-tau50-expe5000-50runs_expeIC_danda.yaml
float const class_centroids[] = {
    0.0001,
    0.01,
    0.40343048418760297,
    0.6576639905929565,
};

uint8_t const class_colors[] = {
    RGB(0,3,0), // green (= error!)
    RGB(2,0,2), // purple
    RGB(2,0,2), // purple
    RGB(0,2,2), // cyan
};
#endif


void set_color_from_lambda(ext_t lambda) {
    ext_t tmp_lambda = lambda;

    // Find closest centroid
    float min_d = 100000.0f;
    uint8_t min_idx = 0;
    for(uint8_t i = 0; i < NCLASSES; i++) {
        float const d = fabs(fabs(tmp_lambda) - class_centroids[i]);
        if(d < min_d) {
            min_d = d;
            min_idx = i;
        }
    }
    // Set color depending on closest centroid
    set_color(class_colors[min_idx]);
}


void set_color_from_s(ext_t s) {
    //ext_t tmp_s = s;
    //ext_t const max_s = initial_s_max_val;
    //ext_t const min_s = -initial_s_max_val;
    ext_t tmp_s = log(fabs(s));
    ext_t const max_s = log(initial_s_max_val);
    ext_t const min_s = log(0.01);

    if(tmp_s < min_s) {
        tmp_s = min_s;
    } else if(tmp_s > max_s) {
        tmp_s = max_s;
    }
    float color_idx = (NRAINBOWCOLORS - 1) * ((float)(tmp_s - min_s) / (float)(max_s - min_s));
    if(color_idx < 0)
        color_idx = 0;
    set_color(colors[(uint8_t)color_idx]);
}

void set_color_from_signs(ext_t s) {
#ifdef ENABLE_INIT_REMOVE_SUM_S
    if(mydata->current_it == 0) {
        if(s <= 0) {
            set_color(RGB(3,0,3));
        } else if(s > 0) {
            set_color(RGB(0,3,0));
        }
    } else {
        if(s <= 0) {
            set_color(RGB(0,0,2));
        } else if(s > 0) {
            set_color(RGB(3,0,0));
        }
    }
#elif defined(ENABLE_PRE_DIFFUSION)
    if(mydata->curr_diff->type == PRE_DIFFUSION_TYPE) {
        if(s <= 0) {
            set_color(RGB(1,1,1));
        } else if(s > 0) {
            set_color(RGB(1,1,1));
        }
    } else {
        if(s <= 0) {
            set_color(RGB(0,0,2));
        } else if(s > 0) {
            set_color(RGB(3,0,0));
        }
    }
#else
    if(s <= 0) {
        set_color(RGB(0,0,2));
    } else if(s > 0) {
        set_color(RGB(3,0,0));
    } /* else if(s == 0) {
        set_color(colors[(int)(NRAINBOWCOLORS/2)-1]);
    } */
#endif
}

void set_color_from_nb_neighbours() {
    uint8_t const nb_neighbours = mydata->nb_neighbors;
    if(nb_neighbours == 0) {
 //       printf("set_color_from_nb_neighbours: nb_neighbours: %d\n", nb_neighbours);
        return;
    }
    float color_idx = (NRAINBOWCOLORS - 1) * ((float)(nb_neighbours - 0) / (float)(MAXN - 0));
    set_color(colors[(uint8_t)color_idx]);
    //printf("set_color_from_nb_neighbours: nb_neighbours: %d   color_idx: %f  tmp:%f \n", nb_neighbours, color_idx, ((float)(nb_neighbours - 0) / (float)(MAXN - 0)));
}


// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
