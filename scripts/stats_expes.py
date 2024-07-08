#!/usr/bin/env python3

import numpy as np
import plots
import sklearn
import sklearn.metrics
import os

expe_names = [
    "2022-09-15_15-45-26.mp4",
    "2022-09-16_10-38-16.mp4",
    "2022-09-16_12-42-32.mp4",

    "2022-07-21_14-16-04.mp4",
    "2022-07-21_15-47-12.mp4",
    "2022-07-22_10-18-21.mp4",
    "2022-07-22_11-50-10.mp4",
    "2022-07-22_13-33-07.mp4",
    "2022-07-22_15-25-54.mp4",
    "2022-07-22_17-06-29.mp4",
    "2022-09-19_10-41-44.mp4",
    "2022-09-19_12-14-47.mp4",
    "2022-09-19_13-43-33.mp4",
    "2022-09-19_15-24-21.mp4",
    "2022-09-19_16-58-24.mp4",
]

y_true_pairs = np.full((len(expe_names), 2), (0, 1))
y_true = y_true_pairs.flatten()

y_expe_pairs = np.array([
    (0, 0),
    (0, 1),
    (1, 1),

    (0, 1),
    (0, 1),
    (0, 1),
    (1, 1),
    (0, 0),
    (0, 1),
    (0, 0),
    (1, 1),
    (0, 1),
    (0, 0),
    (0, 1),
    (1, 1)
])
y_expe = y_expe_pairs.flatten()


############### MAIN ############### {{{1
if __name__ == "__main__":
    stats_per_arena = {'disk': {}, 'annulus': {}}

    # Print classification report
    print(sklearn.metrics.classification_report(y_true, y_expe, target_names=stats_per_arena.keys()))
    # Compute stats
    stats = {}
    stats['accuracy'] = sklearn.metrics.accuracy_score(y_true, y_expe)
    stats['f1'] = sklearn.metrics.f1_score(y_true, y_expe, average='macro')
    stats['precision'] = sklearn.metrics.precision_score(y_true, y_expe, average='macro')
    stats['recall'] = sklearn.metrics.recall_score(y_true, y_expe, average='macro')
    stats['confusion'] = sklearn.metrics.confusion_matrix(y_true, y_expe, normalize='true')
    print(stats)

    # Make confusion matrix
    data = {'stats': stats, 'stats_per_arena': stats_per_arena}
#    colors_dict = {'disk': '#3fbfbf', 'annulus': '#bf3fbf'}
    arena_names = list(data['stats_per_arena'].keys())
#    colors = [colors_dict[x] for x in arena_names]
#    arena_icons = [os.path.join("arenas", f"icon_{a}.png") for a in arena_names]
    colors, arena_icons = plots.get_colors_and_icons(arena_names, plots.colors_dict)

    output_dir = os.path.join("figs", "expe_stationary")
    os.makedirs(output_dir, exist_ok=True)
    plots.confusion_mat_plot(None, data, output_dir, colors=colors, arena_icons=arena_icons)




# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
