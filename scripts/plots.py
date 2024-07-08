#!/usr/bin/env python3

"""TODO"""


import numpy as np
import os
import yaml
import copy
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import scipy

from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
import tessellation
import networkx as nx
import distinctipy

from statannotations.Annotator import Annotator
import scipy
from scipy.stats import mannwhitneyu, normaltest

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r''.join([
        r'\usepackage{amsmath}',
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{helvet}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        r"\usepackage[helvet]{sfmath}",
        r"\everymath={\sf}",
        r'\centering',
        ]))

from runKilombo import *


############### TS PLOTS ############### {{{1

def change_img_color(img, color):
    if type(color) == str:
        c = matplotlib.colors.to_rgb(color)
    else:
        c = color
    img[..., :3] = c[:3]


def plot_line_all_arenas_in_one_plot(x, ys, output_filename, ylim=None, ylabel=r"Final $\lambda_2$", colors = [], arena_icons=[], max_it=None):
    """ x: Sequence of all x values to plot.
        ys: List of all y Sequences for each arena.
        output_filename: filename of the output file
    """
    sns.set(font_scale = 1.3)
    sns.set_style("ticks")
    nb_arenas = len(ys)
    #colors = cmap(np.linspace(0., 1., nb_arenas))
    if len(colors) == 0:
        colors = plt.cm.jet(np.linspace(0., 1., nb_arenas))
    fig, ax = plt.subplots(1, 1, figsize=(4.*scipy.constants.golden, 4.))

    x_ = np.array(x)
    if max_it is not None:
        max_xy = np.where(x_ > max_it)[0]
        if len(max_xy) > 0:
            max_xy = max_xy[0]
            x_ = x_[:max_xy]
        else:
            max_it = None

    last_y_mean = []
    for i,(y, color) in enumerate(zip(ys, colors)):
        y_ = np.array(y).T if max_it is None else np.array(y).T[:max_xy]
        ymean = np.mean(y_, axis = 1)
        last_y_mean.append(ymean[-1])
        ystd = np.std(y_, axis = 1)
        ystd_min = ymean - ystd
        ystd_max = ymean + ystd
        ax.plot(x_, ymean, color=color)
        ax.fill_between(x_, ystd_min, ystd_max, color=color, alpha=0.5)

    plt.xlabel(r"Iterations $\xi$")
    plt.ylabel(ylabel)

    if ylim != None:
        plt.ylim(ylim)

    if len(arena_icons) > 0:
        for i,(y, color, icon) in enumerate(zip(last_y_mean, colors, arena_icons)):
            xy = (x_[-1], y)
            #xy = (0.5, 0.3)
            #ax.plot(xy[0], xy[1], ".r")

            #print(f"DEBUG icons: {xy} {color} {icon}")
            arr_img = plt.imread(icon)
            change_img_color(arr_img, color)
            imagebox = OffsetImage(arr_img, zoom=0.23)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, xy,
                                #xybox=(120., -80.),
                                xybox=(0.97, xy[1]),
                                xycoords='data',
                                #boxcoords="offset points",
                                #pad=0.5
                                boxcoords=("axes fraction", "data"),
                                box_alignment=(0., 0.5),
                                frameon=False
                                )
            ax.add_artist(ab)

    sns.despine(left=False, bottom=False)
    plt.subplots_adjust(left=0.140, bottom=0.150, right=0.980, top=0.970, wspace=0, hspace=0)
    plt.savefig(output_filename)
    plt.close()



def all_plots_ts(config, data, output_dir, colors, arena_icons, class_names=None):
    simulation_time = config['simulationTime']
    kiloticks_iteration = config.get('kiloticks_start_it_waiting_time', 930) + config.get('kiloticks_end_it_waiting_time', 0) + config['kiloticks_randow_walk'] + config['kiloticks_handshake'] + config['kiloticks_diffusion'] * 2 + config['kiloticks_collective_avg_lambda'] + config['kiloticks_collective_avg_avg_lambda']
    total_kiloticks = 31 * simulation_time
    nb_iterations = (total_kiloticks - config['kiloticks_initial_random_walk']) / kiloticks_iteration
    presim_duration = config['kiloticks_initial_random_walk'] + kiloticks_iteration
    index_end_1st_it = int(np.ceil(presim_duration / total_kiloticks * (simulation_time / config['timeStep'] / config['stateFileSteps'])))
    nb_entries = list(data['data_per_arena'].values())[0][0]['avg_lambda'].shape[0]
    if index_end_1st_it >= nb_entries:
        index_end_1st_it = nb_entries - 1

    data_per_arena = data['data_per_arena']

    # Only take into account relevant classes
    if class_names is not None:
        orig_names = data_per_arena.keys()
        data_per_arena = {k:v for k,v in data_per_arena.items() if k in class_names}

    ys = [np.array([r["avg_lambda"][index_end_1st_it:] for r in a]) for a in data_per_arena.values()]
    x = np.linspace(0, nb_iterations, len(ys[0][0]))
    x = [int(a) for a in x]
    plot_line_all_arenas_in_one_plot(x, ys, os.path.join(output_dir, "all_arenas_final_lambda.pdf"), ylabel=r"Final $\lambda_2$", colors=colors, arena_icons=arena_icons)
    plot_line_all_arenas_in_one_plot(x, ys, os.path.join(output_dir, "all_arenas_final_lambda_15it.pdf"), ylabel=r"Final $\lambda_2$", colors=colors, arena_icons=arena_icons, max_it=15)



def plot_line_fstit_s(config, data, output_dir):
    #sns.set(font_scale = 2.4)
    sns.set(font_scale = 2.8)
    sns.set_style("ticks")

    # Find beginning and end of 1st diffusion
    ref_t = list(data['data_per_arena'].values())[0][0]["t"][:,0]
    start_diff_idx = np.where(ref_t > 0.)[0][0]
    end_diff_idx = np.where(ref_t[start_diff_idx:] == 0.)[0]
    if len(end_diff_idx) == 0:
        end_diff_idx = len(ref_t) - 1
        last_val_diff = ref_t[end_diff_idx]
    else:
        end_diff_idx = end_diff_idx[0] + start_diff_idx - 1
        last_val_diff = ref_t[end_diff_idx]
    end_diff_idx = np.where(ref_t[start_diff_idx:end_diff_idx] == last_val_diff)[0][0] + start_diff_idx

    # Find number of diffusion steps
    diff_steps = config['kiloticks_diffusion'] / config['kiloticks_diffusion_it']

    arenas_names = list(data['data_per_arena'].keys())
    ts1 = [np.array(a[0]["t"])[start_diff_idx:end_diff_idx] for a in data['data_per_arena'].values()]
    r1 = [(np.array(a[0]["t"])[start_diff_idx:end_diff_idx,0] * diff_steps / last_val_diff).astype(int) for a in data['data_per_arena'].values()]
    xs1 = [np.array(a[0]["s"])[start_diff_idx:end_diff_idx] for a in data['data_per_arena'].values()]
    ys = [np.array([r["avg_lambda"] for r in a]) for a in data['data_per_arena'].values()]

    dot_x_v2s = []
    #dot_x_v2_v2s = []
    for i, arena in enumerate(arenas_names):
        points = np.array([data['data_per_arena'][arena][0]["x_position"], data['data_per_arena'][arena][0]["y_position"]])
        g = tessellation.create_graph(points[:, start_diff_idx].T, config['commsRadius'])
        cc_idx = max(nx.connected_components(g), key=len) # Get largest connected component indices
        cc_sg = g.subgraph(cc_idx)
        v2 = nx.fiedler_vector(cc_sg, None, True)
        dot = np.dot(xs1[i][:,list(cc_idx)], v2)
        dot_x_v2 = np.abs(dot)
        #dot_x_v2_v2 = np.linalg.norm( (np.vstack(dot) * np.vstack(v2).T), axis=1)
        dot_x_v2s.append(dot_x_v2)
        #dot_x_v2_v2s.append(dot_x_v2_v2)
        #print(dot_x_v2)


    def create_fig(absx, t, dot_x_v2, output_file, data_type="absx", ylim=None):
        normx = np.linalg.norm(absx, axis=1)
        #t = np.arange(0, absx.shape[0]*2, 2)

        # Find proportion of ignored steps from config
        idx_burnin = int(len(t) * config['kiloticks_diffusion_burnin'] / config['kiloticks_diffusion'])

        if data_type == "norm":
            # Estimate mean lambda (skip the first 1/6 part of the ts to be sure to have a straight line in log scale)
            #start_i_mean_lambda = normx.shape[0] // 6 ** 2
            mean_lambda = -(np.log(normx[-1]) - np.log(normx[idx_burnin])) / ((t[-1] - t[idx_burnin]) / config['inv_tau'])
            rot_lambda = np.rad2deg(np.arctan2(normx[-1] - normx[idx_burnin], ((t[-1] - t[idx_burnin])) ))

        elif data_type == "dot":
            mean_lambda = -(dot_x_v2[-1] - dot_x_v2[idx_burnin]) / ((t[-1] - t[idx_burnin]) / config['inv_tau'])

        fig, ax = plt.subplots(1, 1, figsize=(5.1*scipy.constants.golden, 5.1))
        plt.tick_params(axis='y', which='minor')

        if ylim is not None:
            plt.ylim(ylim)

        if data_type == "norm":
            # Burn-in zone
            plt.axvline(x=t[idx_burnin], color='k', alpha=0.5)
            ax.axvspan(0, t[idx_burnin], alpha=0.08, color='red')
            ax.annotate("burn\nin", fontsize=30, xy = (0.080, 0.025), xycoords="axes fraction", xytext=(2,10), ha='center', va='baseline', textcoords="offset points")
            plt.plot(t[:idx_burnin+1], normx[:idx_burnin+1], "k:", linewidth=4.0)

            # Draw plot
            plt.plot(t[idx_burnin:], normx[idx_burnin:], "k-", linewidth=4.0)
            plt.xlim(0, np.max(t))

            #plt.ylabel(r"$norm(s^{t})$")
            plt.ylabel(r"$|| (\mathbf{s} \cdot \mathbf{v}_2) \mathbf{v}_2 ||$")
            plt.yscale('log')
            plt.minorticks_on()

            # Annotate ts to add mean lambda value
            #ax.annotate(r"mean $\lambda = " + f"{mean_lambda:.3f}$", fontsize=23, xy = (t[len(t)//2]-30, normx[len(normx)//2]), xytext=(2,10), ha='center', va='baseline', textcoords="offset points", rotation_mode="anchor", rotation=angle_screen)
            ax.annotate(rf"$\lambda_2 = {mean_lambda:.3f} " + r"\mathrm{m}^2$", fontsize=43, xy = (0.75, 0.75), xycoords="figure fraction", xytext=(2,10), ha='center', va='baseline', textcoords="offset points")
            ax.annotate(r"$\textbf{s}^n \approx e^{-\lambda_2 n c \tau}$", fontsize=43, xy = (0.54, 0.25), xycoords="figure fraction", xytext=(2,10), ha='center', va='baseline', textcoords="offset points")


        elif data_type == "absx":
            # Generate plot colors
            colors = distinctipy.get_colors(absx.shape[1], colorblind_type="Deuteranomaly")

            # Burn-in zone
            plt.axvline(x=t[idx_burnin], color='k', alpha=0.5)
            ax.axvspan(0, t[idx_burnin], alpha=0.08, color='red')
            #ax.annotate(r"burn-in", fontsize=21, xy = (0.080, 0.05), xycoords="axes fraction", xytext=(2,10), ha='center', va='baseline', textcoords="offset points")
            ax.annotate("burn\nin", fontsize=30, xy = (0.080, 0.025), xycoords="axes fraction", xytext=(2,10), ha='center', va='baseline', textcoords="offset points")
            for v,c in zip(absx.T, colors):
                #plt.plot(t, v, color=c)
                plt.plot(t[:idx_burnin+1], v[:idx_burnin+1], ":", color=c, linewidth=1.0)

            # Draw plot
            for v,c in zip(absx.T, colors):
                #plt.plot(t, v, color=c)
                plt.plot(t[idx_burnin:], v[idx_burnin:], color=c)
            plt.xlim(0, np.max(t))

            #plt.ylabel(r"$abs(s_{i}^t)$")
            plt.ylabel(r"$| s_{i}^n |$")
            plt.yscale('log')
            plt.minorticks_on()

        elif data_type == "dot":
            plt.plot(t[:idx_burnin+1], dot_x_v2[:idx_burnin+1], "k:", linewidth=4.0)
            plt.axvline(x=t[idx_burnin], color='k', alpha=0.5)
            plt.plot(t[idx_burnin:], dot_x_v2[idx_burnin:], "k-", linewidth=4.0)

            plt.ylabel(r"$|| (\mathbf{s} \cdot \mathbf{v}_2) \mathbf{v}_2 ||$")
            ax.annotate(r"$\lambda_2 = " + f"{mean_lambda:.4f}$", fontsize=35, xy = (0.65, 0.8), xycoords="figure fraction", xytext=(2,10), ha='center', va='baseline', textcoords="offset points")

        plt.xlabel(r"Diffusion step $n$")
        #plt.xlabel("Diffusion step")
        #plt.xlabel(r"Diffusion time $t \ [\mathrm{s}]$")
        #plt.xlabel(r"Diffusion time $t \ [\tau$\ units$]$")
        #ax.annotate(r'$\times \tau$', fontsize=35, xy=(0.95, 0.02), xycoords="figure fraction", xytext=(2,10), ha='center', va='baseline', textcoords="offset points")

        #plt.tight_layout()
        #plt.subplots_adjust(left=0.185, bottom=0.190, right=0.990, top=0.95, wspace=0, hspace=0)
        #plt.subplots_adjust(left=0.23, bottom=0.22, right=0.990, top=0.95, wspace=0, hspace=0)
        plt.subplots_adjust(left=0.20, bottom=0.20, right=0.995, top=0.99, wspace=0, hspace=0)
        plt.savefig(os.path.join(output_dir, output_file))
        plt.close()

    for i, arena in enumerate(arenas_names):
        absx1 = np.abs(xs1[i])
        create_fig(absx1, r1[i], dot_x_v2s[i], f"fstit_abss_{arena}-1.pdf", "absx", ylim=[5e-4, 2e0])
        create_fig(absx1, r1[i], dot_x_v2s[i], f"fstit_norms_{arena}-1.pdf", "norm", ylim=[1e-1, 5e0])



############### OTHER PLOTS ############### {{{1


def smooth_iqr(data, q1 = 0.25, q2 = 0.75):
    assert(q2 > q1)
    data = np.array(data)
    quantile1 = np.quantile(data, q1, axis=0)
    quantile2 = np.quantile(data, q2, axis=0)
    iqr = quantile2 - quantile1
    min_val = quantile1 - 1.5 * iqr
    max_val = quantile2 + 1.5 * iqr
    if len(data.shape) > 1:
        data = data[np.logical_and( np.all(data>=min_val, axis=-1), np.all(data<=max_val, axis=-1) )]
    else:
        data = data[np.logical_and(data>min_val, data<max_val)]
    return data


def plot_line_all_arenas(x, ys, output_filename, titles = None, color='g'):
    """ x: Sequence of all x values to plot.
        ys: List of all y Sequences for each arena.
        output_filename: filename of the output file
    """
    sns.set(font_scale = 1.0)
    sns.set_style("ticks")

    nb_subplots = len(ys)
    #fig, axs = plt.subplots(1, nb_subplots, figsize=(nb_subplots * 15.0, 5.0))
    fig, axs = plt.subplots(1, nb_subplots, figsize=(nb_subplots * 5.0, 5.0))
    if nb_subplots == 1:
        axs = [axs]

    for i,(y,ax) in enumerate(zip(ys, axs)):
        y_ = np.array(y).T
        ymean = np.mean(y_, axis = 1)
        ymedian = np.median(y_, axis = 1)
        ymin = np.min(y_, axis = 1)
        ymax = np.max(y_, axis = 1)
        y10 = np.quantile(y_, 0.10, axis = 1)
        y25 = np.quantile(y_, 0.25, axis = 1)
        y40 = np.quantile(y_, 0.40, axis = 1)
        y60 = np.quantile(y_, 0.60, axis = 1)
        y75 = np.quantile(y_, 0.75, axis = 1)
        y90 = np.quantile(y_, 0.90, axis = 1)

        ax.plot(x, ymean, 'k-')
        ax.plot(x, ymedian, 'y--')
        ax.fill_between(x, ymin, ymax, color=color, alpha=0.3)
        ax.fill_between(x, y10, y90, color=color, alpha=0.5)
        ax.fill_between(x, y25, y75, color=color, alpha=0.7)
        ax.fill_between(x, y40, y60, color=color, alpha=1.0)

        if titles is not None and len(titles) > i:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()



#def plot_violin_all_arenas(xs, output_filename, titles = None, color='o'):
def plot_violin_all_arenas(xs, output_filename, titles = None, colors=[], arena_icons=[], ylim=[0., 1.], ylabel="", ref_vals=None, annotate_pvalues=False):
    """ xs: List of all y Sequences for each arena.
        output_filename: filename of the output file
    """
    sns.set(font_scale = 1.6)
    #sns.set_style("ticks")
    sns.set_style("white")

    # Set custom color palette
    sns.set_palette(sns.color_palette(colors))

    #fig, ax = plt.subplots(figsize=(4.0*scipy.constants.golden, 4.0))
    fig, ax = plt.subplots(figsize=(2.5, 2.5*scipy.constants.golden))
    v = sns.violinplot(data=xs, ax=ax)
    v.set_ylim(ylim)

    plt.ylabel(ylabel, labelpad=3)

    # Draw the icons
    nb_arenas = len(xs)
    ax = plt.gca()
    ticks = np.arange(0, nb_arenas, 1) + 0.5
    for i,(t, color, icon) in enumerate(zip(ticks, colors, arena_icons)):
        xy_c = (t / nb_arenas, 0)
        #xy = (0.5, 0.3)
        #ax.plot(xy[0], xy[1], ".r")

        #print(f"DEBUG icons: {xy} {color} {icon}")
        arr_img = plt.imread(icon)
        change_img_color(arr_img, color)
        imagebox = OffsetImage(arr_img, zoom=0.35)
        imagebox.image.axes = ax
        ab2 = AnnotationBbox(imagebox, xy_c,
                #xybox=(xy_c[0], nb_arenas + yshift),
                            xybox=(xy_c[0], (-0.11) * ylim[1]),
                            xycoords='data',
                            boxcoords=("axes fraction", "data"),
                            box_alignment=(0.5, 0.0),
                            frameon=False
                            )
        ax.add_artist(ab2)

    if ref_vals is not None:
        for i,val in enumerate(ref_vals):
            #ax.axhline(y = val, xmin = i, xmax = i+0.5)
            ax.axhline(y = val, xmin = i-0.5, xmax = i+0.5, linestyle = '-', color='r', lw=4)
            #ax.axhline(y = val)

    if annotate_pvalues:
        annotator = Annotator(ax, [[0, 1]], data=xs)
        annotator.configure(test='Mann-Whitney', text_format='simple', comparisons_correction="Bonferroni", fontsize=10, loc='outside')
        plt.rc('text', usetex=False)
        annotator.apply_and_annotate()
        plt.rc('text', usetex=True)

    if titles is not None:
        v.set(xticklabels = titles)
    else:
        v.set(xticklabels = [""] * nb_arenas)
    sns.despine(left=False, bottom=False)
    #plt.tight_layout()
    if len(ylabel) == 0:
        plt.subplots_adjust(left=0.25, bottom=0.13, right=0.93, top=0.97, wspace=0, hspace=0)
    else:
        if annotate_pvalues:
            plt.subplots_adjust(left=0.37, bottom=0.10, right=0.99, top=0.90, wspace=0, hspace=0)
        else:
            plt.subplots_adjust(left=0.37, bottom=0.10, right=0.99, top=0.97, wspace=0, hspace=0)
    plt.savefig(output_filename)
    plt.close()



def plot_embedding_v_all_arenas(data_per_arena, output_filename, color='r', max_runs = 10, key_v = "last_v"):
    sns.set(font_scale = 1.0)
    sns.set_style("ticks")

    arenas_names = list(data_per_arena.keys())
    nb_arenas = len(arenas_names)
    nb_runs = min(len(data_per_arena[arenas_names[0]]), max_runs)

    # Create figure
    fig, axs = plt.subplots(nb_arenas, nb_runs, figsize=(nb_runs * 4.0, nb_arenas * 4.0))
    if nb_arenas == 1:
        axs = np.array([axs])
    if nb_runs == 1:
        axs = np.array([axs]).T

    # Plot embedding (v1,v2) for the first ``max_runs`` run
    for i, arena_n in enumerate(arenas_names):
        a = data_per_arena[arena_n]
        for j, run in enumerate(a[:max_runs]):
            v = run[key_v]
            smoothed_v = smooth_iqr(v)
            axs[i, j].plot(smoothed_v[:,0], smoothed_v[:,1], color + 'o')
            if j == 0:
                axs[i, j].set_ylabel(arena_n)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()



def all_plots(config, data, output_dir):
    simulation_time = config['simulationTime']
    arena_names = list(data['data_per_arena'].keys())
    colors, arena_icons = get_colors_and_icons(arena_names, colors_dict)

    ######## base data/stats ########

    # Plot evolution of avg_lambda wrt to time for all arenas
    ys = [np.array([r["avg_lambda"] for r in a]) for a in data['data_per_arena'].values()]
    x = np.linspace(0, simulation_time, len(ys[0][0]))
    plot_line_all_arenas(x, ys, os.path.join(output_dir, "avg_lambda.pdf"), color='g', titles=arena_names)

    # Violin plot of the distribution of final avg_lambda for all arenas
    xs = [smooth_iqr(a['last_avg_lambda']) for a in data['stats_per_arena'].values()]
    #plot_violin_all_arenas(xs, os.path.join(output_dir, "last_avg_lambda-violin.pdf"), titles=arena_names)
    #plot_violin_all_arenas(xs, os.path.join(output_dir, "last_avg_lambda-violin.pdf"), titles=None, colors=colors, arena_icons=arena_icons, ylabel=r"consensus$\lambda$")
    plot_violin_all_arenas(xs, os.path.join(output_dir, "last_avg_lambda-violin.pdf"), titles=None, colors=colors, arena_icons=arena_icons, ylabel=r"Final $\lambda_2$")

    if len(xs) == 2 and 'disk' in data['stats_per_arena'].keys() and 'annulus' in data['stats_per_arena'].keys():
        xs = np.array([a['last_avg_lambda'] for a in data['stats_per_arena'].values()])
        xs_normalized = (xs - xs.min()) / (xs[0].mean() - xs.min())
        ref_vals = np.array([1.0, 0.242908]) # Ratio values from Continuous Laplacian
        plot_violin_all_arenas(xs_normalized.tolist(), os.path.join(output_dir, "last_avg_lambda-normalized-violin.pdf"), titles=None, colors=colors, arena_icons=arena_icons, ylabel=r"Normalized Final $\lambda_2$", ref_vals = ref_vals, ylim=(0.0, 1.5), annotate_pvalues=True)

    # Plot of the distribution of degrees
    try:
        plots_distrib_degree(config, data, output_dir)
    except Exception as e:
        pass

    # Check if there are extended data/stats
    if "s" not in list(data['data_per_arena'].values())[0][0].keys():
        return


    ######## extended data/stats ########

#    # Plot evolution of the sum of all s wrt to time for all arenas
#    ys = [np.array([r["sum_s"] for r in a]) for a in data['data_per_arena'].values()]
#    x = np.linspace(0, simulation_time, len(ys[0][0]))
#    plot_line_all_arenas(x, ys, os.path.join(output_dir, "sum_s.pdf"), color='g', titles=arena_names)
#
#    # Plot evolution of the t of agent 0 wrt to time for all arenas
#    ys = [np.array([a[0]["t"][:,0]]) for a in data['data_per_arena'].values()]
#    x = np.linspace(0, simulation_time, len(ys[0][0]))
#    plot_line_all_arenas(x, ys, os.path.join(output_dir, "agent0_t.pdf"), color='g', titles=arena_names)
#
#    # Plot evolution of the s of agent 0 wrt to time for all arenas
#    ys = [np.array([a[0]["s"][:,0]]) for a in data['data_per_arena'].values()]
#    x = np.linspace(0, simulation_time, len(ys[0][0]))
#    plot_line_all_arenas(x, ys, os.path.join(output_dir, "agent0_s.pdf"), color='g', titles=arena_names)
#    # Plot evolution of the s of agent 0 wrt to time for all arenas only during diffusion 1
#    nanvect = np.full(len(ys[0][0]), np.nan)
#    curr_behav = [np.array([r["mean_current_behavior"] for r in a]) for a in data['data_per_arena'].values()]
#    ys1 = [np.where(behav == 1, y, nanvect) for y, behav in zip(ys, curr_behav)]
#    plot_line_all_arenas(x, ys1, os.path.join(output_dir, "agent0_s1.pdf"), color='g', titles=arena_names)
#
#    # Plot evolution of the s of agent 1 wrt to time for all arenas
#    ys = [np.array([a[0]["s"][:,1]]) for a in data['data_per_arena'].values()]
#    x = np.linspace(0, simulation_time, len(ys[0][0]))
#    plot_line_all_arenas(x, ys, os.path.join(output_dir, "agent1_s.pdf"), color='g', titles=arena_names)
#    # Plot evolution of the s of agent 1 wrt to time for all arenas only during diffusion 1
#    ys1 = [np.where(behav == 1, y, nanvect) for y, behav in zip(ys, curr_behav)]
#    plot_line_all_arenas(x, ys1, os.path.join(output_dir, "agent1_s1.pdf"), color='g', titles=arena_names)
#
#
#    # Plot evolution of log(abs(s)) of agent 0 wrt to time for all arenas
#    ys = [np.log(np.abs(np.array([a[0]["s"][:,0]]))) for a in data['data_per_arena'].values()]
#    x = np.linspace(0, simulation_time, len(ys[0][0]))
#    plot_line_all_arenas(x, ys, os.path.join(output_dir, "agent0_logabss.pdf"), color='g', titles=arena_names)
#    # Plot evolution of log(abs(s)) of agent 0 wrt to time for all arenas during diffusion 1
#    ys1 = [np.where(behav == 1, y, nanvect) for y, behav in zip(ys, curr_behav)]
#    plot_line_all_arenas(x, ys1, os.path.join(output_dir, "agent0_logabss1.pdf"), color='g', titles=arena_names)
#    # Plot evolution of log(abs(s)) of agent 1 wrt to time for all arenas
#    ys = [np.log(np.abs(np.array([a[0]["s"][:,1]]))) for a in data['data_per_arena'].values()]
#    x = np.linspace(0, simulation_time, len(ys[0][0]))
#    plot_line_all_arenas(x, ys, os.path.join(output_dir, "agent1_logabss.pdf"), color='g', titles=arena_names)
#    # Plot evolution of log(abs(s)) of agent 1 wrt to time for all arenas during diffusion 1
#    ys1 = [np.where(behav == 1, y, nanvect) for y, behav in zip(ys, curr_behav)]
#    plot_line_all_arenas(x, ys1, os.path.join(output_dir, "agent1_logabss1.pdf"), color='g', titles=arena_names)



#    # Plot evolution of mean_current_behavior wrt to time for all arenas
#    ys = [np.array([r["mean_current_behavior"] for r in a]) for a in data['data_per_arena'].values()]
#    x = np.linspace(0, simulation_time, len(ys[0][0]))
#    plot_line_all_arenas(x, ys, os.path.join(output_dir, "mean_current_behavior.pdf"), color='r', titles=arena_names)
#
#    # Plot evolution of mean_diffusion_valid wrt to time for all arenas
#    ys = [np.array([r["mean_diffusion_valid"] for r in a]) for a in data['data_per_arena'].values()]
#    x = np.linspace(0, simulation_time, len(ys[0][0]))
#    plot_line_all_arenas(x, ys, os.path.join(output_dir, "mean_diffusion_valid.pdf"), color='r', titles=arena_names)

    # Plot evolution of lambda wrt to time for all arenas
    ys = [np.array([r["lambda"] for r in a]) for a in data['data_per_arena'].values()]
    x = np.linspace(0, simulation_time, len(ys[0][0]))
    plot_line_all_arenas(x, ys, os.path.join(output_dir, "lambda.pdf"), color='g', titles=arena_names)

    # Plot evolution of autocorrelations of lambda wrt to time for all arenas
    ys = [a['mean_autocorr_lambda'] for a in data['stats_per_arena'].values()]
    x = np.linspace(0, simulation_time, len(ys[0][0]))
    plot_line_all_arenas(x, ys, os.path.join(output_dir, "autocorr-lambda.pdf"), color='b', titles=arena_names)

    # Violin plot of the distribution of final lambda for all arenas
    xs = [smooth_iqr(a['last_lambda']) for a in data['stats_per_arena'].values()]
    plot_violin_all_arenas(xs, os.path.join(output_dir, "last_lambda-violin.pdf"), titles=None, colors=colors, arena_icons=arena_icons, ylabel=r"$\lambda$")

    # Violin plot of the distribution of last_mse for all arenas
    #xs = [smooth_iqr(np.concatenate(a['last_mse_0'], a['last_mse_1'], a['last_mse_2'])) for a in data['data_per_arena'].values()]
    #xs = [smooth_iqr(np.array([np.concatenate(r['last_mse_0'], r['last_mse_1'], r['last_mse_2']) for r in a]).flatten()) for a in data['data_per_arena'].values()]
    xs = [smooth_iqr(np.array([r['last_mse_0'] for r in a]).flatten()) for a in data['data_per_arena'].values()]
    #plot_violin_all_arenas(xs, os.path.join(output_dir, "last_avg_lambda-violin.pdf"), titles=arena_names)
    plot_violin_all_arenas(xs, os.path.join(output_dir, "last_mse-violin.pdf"), titles=None, colors=colors, arena_icons=arena_icons, ylim=[0., 10.], ylabel=r"MSE")


    ## Plot evolution of avg_lambda wrt to time for all arenas
    #ys = [np.array([r["avg_lambda"] for r in a]) for a in data['data_per_arena'].values()]
    #x = np.linspace(0, simulation_time, len(ys[0][0]))
    #plot_line_all_arenas(x, ys, os.path.join(output_dir, "avg_lambda.pdf"), color='g', titles=arena_names)

    ## Plot evolution of autocorrelations of avg_lambda wrt to time for all arenas
    #ys = [a['mean_autocorr_avg_lambda'] for a in data['stats_per_arena'].values()]
    #x = np.linspace(0, simulation_time, len(ys[0][0]))
    #plot_line_all_arenas(x, ys, os.path.join(output_dir, "autocorr-avg_lambda.pdf"), color='b', titles=arena_names)

    ## Violin plot of the distribution of final avg_lambda for all arenas
    #xs = [smooth_iqr(a['last_avg_lambda']) for a in data['stats_per_arena'].values()]
    #plot_violin_all_arenas(xs, os.path.join(output_dir, "last_avg_lambda-violin.pdf"), titles=None, colors=colors, arena_icons=arena_icons)

#    # Violin plot of distribution of intermediate avg_lambda for all arenas (1/8, 2/8, 4/8, 6/8 of budget)
#    try:
#        fst_arena_name = list(data['data_per_arena'].keys())[0]
#        nb_it = len(data['data_per_arena'][fst_arena_name][0]['avg_lambda'])
#        xs = np.array([smooth_iqr([r["avg_lambda"][nb_it//8*1] for r in a]) for a in data['data_per_arena'].values()])
#        plot_violin_all_arenas(xs, os.path.join(output_dir, f"it{nb_it//8*1}_avg_lambda-violin.pdf"), titles=None, colors=colors, arena_icons=arena_icons)
#        xs = np.array([smooth_iqr([r["avg_lambda"][nb_it//8*2] for r in a]) for a in data['data_per_arena'].values()])
#        plot_violin_all_arenas(xs, os.path.join(output_dir, f"it{nb_it//8*2}_avg_lambda-violin.pdf"), titles=None, colors=colors, arena_icons=arena_icons)
#        xs = np.array([smooth_iqr([r["avg_lambda"][nb_it//8*4] for r in a]) for a in data['data_per_arena'].values()])
#        plot_violin_all_arenas(xs, os.path.join(output_dir, f"it{nb_it//8*4}_avg_lambda-violin.pdf"), titles=None, colors=colors, arena_icons=arena_icons)
#        xs = np.array([smooth_iqr([r["avg_lambda"][nb_it//8*6] for r in a]) for a in data['data_per_arena'].values()])
#        plot_violin_all_arenas(xs, os.path.join(output_dir, f"it{nb_it//8*6}_avg_lambda-violin.pdf"), titles=None, colors=colors, arena_icons=arena_icons)
#    except Exception as e:
#        pass

    try:
        plot_line_fstit_s(config, data, output_dir)
    except Exception as e:
        pass


############### CONFUSION MATRICES PLOTS ############### {{{1


def change_order_confusion_mat(confusion, colors, arena_icons, ordered_class_names = ['disk', 'square', 'arrow2', 'star', 'triangle', 'arena8', 'arena6', 'annulus']):
    orig_classes_names = list(data['stats_per_arena'].keys())
    intersect_class_names = set(orig_classes_names) & set(ordered_class_names)
    nb_arenas = len(intersect_class_names)
    #assert(set(orig_classes_names) == set(ordered_class_names))

    orig_coord = {c: i for i, c in enumerate(orig_classes_names)}
    ordered_coord = {c: i for i, c in enumerate(ordered_class_names)}

    def ordered_to_orig(i):
        ordered_c = ordered_class_names[i]
        return orig_coord[ordered_c]

    new_c = np.empty((nb_arenas, nb_arenas))
    new_colors = []
    new_arena_icons = []
    i = 0
    for c in ordered_class_names:
        if c not in intersect_class_names:
            continue
        new_colors.append(colors[ordered_to_orig(i)])
        new_arena_icons.append(arena_icons[ordered_to_orig(i)])
        j = 0
        for d in ordered_class_names:
            if d not in intersect_class_names:
                continue
            new_c[i, j] = confusion[ordered_to_orig(i), ordered_to_orig(j)]
            j += 1
        i += 1

    return new_c, new_colors, new_arena_icons


def confusion_mat_plot(config, data, output_dir, colors=[], arena_icons=[], ordered_class_names=None):

    # Recompute stats?
    if ordered_class_names is not None:
        classes_names = set(data['stats_per_arena'].keys()) & set(ordered_class_names)
        assert(len(classes_names) > 0)
        stats_per_arena = {k:v for k,v in data['stats_per_arena'].items() if k in classes_names}
        data_per_arena = {k:v for k,v in data['data_per_arena'].items() if k in classes_names}
        stats = compute_stats_all_arenas(data_per_arena, stats_per_arena, config, None)
    else:
        stats = data['stats']
        classes_names = list(data['stats_per_arena'].keys())

    if 'confusion' not in stats:
        return
    sns.set_style("ticks")
    confusion = stats['confusion']

    # Reorder classes if needed
    if ordered_class_names is not None:
        classes_names = ordered_class_names
        confusion, colors, arena_icons = change_order_confusion_mat(confusion, colors, arena_icons, ordered_class_names)
        classes_names = ordered_class_names
    nb_arenas = confusion.shape[0]

    if nb_arenas == 2:
        sns.set(font_scale = 2.0)
        cbar_kws = {'label': 'Prop. of instances'}
        annot_kws = {}
    else:
        sns.set(font_scale = 1.4)
        cbar_kws = {'label': 'Prop. of instances'}
        annot_kws = {'fontsize': 13}

    if len(arena_icons) == 0:
        sns.heatmap(confusion, cmap="Greens", xticklabels=classes_names, yticklabels=classes_names, annot=True, fmt="0.2g", linewidths=.5, 
                cbar_kws=cbar_kws, vmin=0., vmax=1.0, annot_kws=annot_kws)
    else:
        sns.heatmap(confusion, cmap="Greens", xticklabels=[], yticklabels=[], annot=True, fmt="0.2g", linewidths=.5, 
                cbar_kws=cbar_kws, vmin=0., vmax=1.0, annot_kws=annot_kws)
        # Define icon colors
        #colors = cmap(np.linspace(0., 1., nb_arenas))
        if len(colors) == 0:
            colors = plt.cm.jet(np.linspace(0., 1., nb_arenas))
        # Draw the icons
        ax = plt.gca()
        ticks = np.arange(0, nb_arenas, 1) + 0.5
        if nb_arenas == 2:
            yshift = 0.24
            xshift = -0.11
        elif nb_arenas == 6:
            yshift = 0.60
            xshift = -0.10
        else:
            yshift = 0.80
            xshift = -0.09
        for i,(t, color, icon) in enumerate(zip(ticks, colors, arena_icons)):
            xy_l = (0, t)
            xy_c = (t / nb_arenas, 0)
            #xy = (0.5, 0.3)
            #ax.plot(xy[0], xy[1], ".r")

            #print(f"DEBUG icons: {xy} {color} {icon}")
            arr_img = plt.imread(icon)
            change_img_color(arr_img, color)
            if nb_arenas == 2:
                imagebox = OffsetImage(arr_img, zoom=0.60)
            else:
                imagebox = OffsetImage(arr_img, zoom=0.35)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, xy_l,
                                #xybox=(120., -80.),
                                #xybox=(-0.06, xy_l[1]),
                                xybox=(xshift, xy_l[1]),
                                xycoords='data',
                                #boxcoords="offset points",
                                #pad=0.5
                                boxcoords=("axes fraction", "data"),
                                box_alignment=(0., 0.5),
                                frameon=False
                                )
            ax.add_artist(ab)
            ab2 = AnnotationBbox(imagebox, xy_c,
                                xybox=(xy_c[0], nb_arenas + yshift),
                                xycoords='data',
                                boxcoords=("axes fraction", "data"),
                                box_alignment=(0.5, 0.0),
                                frameon=False
                                )
            ax.add_artist(ab2)


    if nb_arenas == 2:
        plt.xlabel("Predicted classes", labelpad=36)
        plt.ylabel("Actual classes", labelpad=35)
        plt.subplots_adjust(left=0.13, bottom=0.175, right=0.93, top=0.97, wspace=0, hspace=0)
    elif nb_arenas == 6:
        plt.xlabel("Predicted classes", labelpad=37)
        plt.ylabel("Actual classes", labelpad=32)
        plt.subplots_adjust(left=0.12, bottom=0.170, right=0.93, top=0.97, wspace=0, hspace=0)
    elif nb_arenas == 7:
        plt.xlabel("Predicted classes", labelpad=37)
        plt.ylabel("Actual classes", labelpad=32)
        plt.subplots_adjust(left=0.110, bottom=0.155, right=0.98, top=0.98, wspace=0, hspace=0)
    elif nb_arenas == 8:
        plt.xlabel("Predicted classes", labelpad=32)
        plt.ylabel("Actual classes", labelpad=29)
        plt.subplots_adjust(left=0.11, bottom=0.150, right=0.96, top=0.97, wspace=0, hspace=0)
    else:
        plt.ylabel("Actual classes", labelpad=22)
        plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion.pdf"))
    plt.close()



def plots_distrib_degree(config, data, output_dir, per_run=True):
    sns.set(font_scale = 2.5)
    fig, axes = plt.subplots(1, len(data['stats_per_arena'].keys()), figsize=(30, 6), sharey=True)
    ylim = [0., 0.28]
    distrib_degrees = {k: v['tess_stats']['degrees'] for k,v in data['stats_per_arena'].items()}
    for i, (a, k) in enumerate(zip(range(len(axes)), colors_dict.keys())):
        v = distrib_degrees[k]
        #print(k)
        #g = sns.kdeplot(v, ax=axes[a])

        if per_run:
            K = v.shape[0]
            N = v.shape[1]
            lbl = np.repeat(np.arange(K), N)
            bins = np.arange(0, 21, 1)
            counts = np.array([np.histogram(v[j], bins, density=True)[0] for j in range(K)])
            df = pd.DataFrame({'probability': counts.ravel(), 'degree': np.tile(bins[:-1], K), 'label': np.repeat(np.arange(K), len(bins) - 1)})
            g = sns.barplot(data=df, x='degree', y='probability', ax=axes[a], color = colors_dict[k], width=1.00)
            axes[a].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3))
            axes[a].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

            means_v = v.mean(1)
            skews_v = scipy.stats.skew(v, 1)
            title = f"mean={means_v.mean():.2f}±{means_v.std():.2f}\nskew={skews_v.mean():.2f}±{skews_v.std():.2f}"
            #axes[a].set_title(f"{k.capitalize()}\nmean={means_v.mean():.2f}±{means_v.std():.2f}\nskew={skews_v.mean():.2f}±{skews_v.std():.2f}", fontsize=22)
            #axes[a].set_title(title, fontsize=22)
            plt.text(0.5, 0.9, title, horizontalalignment='center', verticalalignment='center', transform = axes[a].transAxes)

        else:
            g = sns.histplot(v.flatten(), ax=axes[a], stat="probability", discrete=True, color=colors_dict[k])
            mean_v = v.mean()
            skew = scipy.stats.skew(v.ravel())
            #iqr = np.quantile(v, 0.75) - np.quantile(v, 0.25)
            var_v = v.var()
            kurtosis = scipy.stats.kurtosis(v.ravel())
            #axes[a].set_title(f"{k.capitalize()}\nmean={mean_v:.2f} var={var_v:.2f}\nskew={skew:.3f} kurtosis={kurtosis:.3f}", fontsize=22)
            axes[a].set_title(f"mean={mean_v:.2f} var={var_v:.2f}\nskew={skew:.3f} kurtosis={kurtosis:.3f}", fontsize=22)

        plt.ylim(ylim)
        g.vlines(x=v.mean(), ymin=ylim[0], ymax=ylim[1], colors='tab:orange', ls='--')

    #fig.tight_layout(pad=1.0)
    plt.subplots_adjust(left=0.05, bottom=0.16, right=0.999, top=0.99, wspace=0.1, hspace=0)
    plt.savefig(os.path.join(output_dir, "dist_degrees.pdf"))
    plt.close('all')



def plots_reduced_distrib_degree(config, data, output_dir):
    sns.set(font_scale = 1.3)
    ylim = [0., 0.10]
    fig, ax = plt.subplots(1, 1, figsize=(4.0*scipy.constants.golden, 4.0))
    distrib_degrees = {k: v['tess_stats']['degrees'].flatten() for k,v in data['stats_per_arena'].items()}
    v = pd.DataFrame.from_dict({k: distrib_degrees[k] for k in ['disk', 'triangle', 'annulus']})
    skew = {k: scipy.stats.skew(v) for k, v in v.items()}
    means = {k: np.mean(v) for k, v in v.items()}
    labels = [rf"\parbox{{2cm}}{{{k}}} skewness={skew[k]:.2f}" for k in v.keys()]
    colors = {labels[i]: colors_dict[list(v.keys())[i]] for i in range(len(labels))}
    v.columns = labels
    g = sns.kdeplot(v, legend=True, palette=colors, ax=ax)
    plt.ylim(ylim)
    for m, c in zip(means.values(), colors.values()):
        g.vlines(x=m, ymin=ylim[0], ymax=ylim[1], colors=c, ls='--')
    ax.set_xlabel("Degree")
    fig.tight_layout(pad=1.0)
    plt.savefig(os.path.join(output_dir, "reduced_dist_degrees.pdf"))
    plt.close('all')




# Darker color to be visually similar to kilombo images (that have a black arrow in the middle of agents, changing the perceived color)
colors_dict = {
        'disk':     '#439494',      # Cyan
        'square':   '#bd6034',      # Orange
        'arrow2':   '#6c916c',      # Green
        'star':     '#b64949',      # Red
        'triangle': '#8b8b47',      # Gold
#        'arena8':   '#956e6e',      # Brown
#        'arena6':   '#6e6e6e',      # Grey
        'annulusBarred':   '#956e6e',      # Brown
        'annulus':  '#975497',      # Magenta
        }

def get_colors_and_icons(arena_names, colors_d = colors_dict):
    colors = [colors_dict.get(x, "#7f3f7f") for x in arena_names]
    arena_icons = [os.path.join("arenas", f"icon_{a}.png") for a in arena_names]
    return colors, arena_icons


############### MAIN ############### {{{1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configPath', type=str, default='', help = "Path of the config file")
    parser.add_argument('-i', '--inputPath', type=str, default='data.p', help = "Path of the data path")
    parser.add_argument('-o', '--outputDir', type=str, default=".", help = "Directory of the resulting plot files")
    args = parser.parse_args()

    output_dir = args.outputDir
    # Load data
    with open(args.inputPath, "rb") as f:
        data = pickle.load(f)

    # Load config
    if len(args.configPath) > 0:
        config = yaml.safe_load(open(args.configPath))
    else:
        config = data['base_config']

    # Find all icons
    arena_names = list(data['stats_per_arena'].keys())
    colors, arena_icons = get_colors_and_icons(arena_names, colors_dict)

    if len(arena_names) > 2:
        ordered_class_names = list(colors_dict.keys())
    else:
        ordered_class_names = None

    # Make all plots
    #confusion_mat_plot(config, data, output_dir, colors=colors, arena_icons=arena_icons)
    confusion_mat_plot(config, data, output_dir, colors=colors, arena_icons=arena_icons, ordered_class_names = ordered_class_names)
    all_plots_ts(config, data, output_dir, colors=colors, arena_icons=arena_icons, class_names = ordered_class_names)
    all_plots(config, data, output_dir)

    #plot_line_fstit_s(config, data, output_dir)

# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
