#!/usr/bin/env python3

"""TODO"""


import numpy as np
import warnings
import os
import pathlib
import shutil
import datetime
import subprocess
import traceback
#import sys
#import glob
import json
import yaml
import copy
import pickle
import sklearn
import sklearn.metrics
import scipy

import tessellation
import multiprocessing

import matplotlib.pyplot as plt


# From https://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
def estimated_autocorrelation(x):
    xa = np.array(x)
    n = len(xa)
    variance = xa.var()
    xa = xa-xa.mean()
    r = np.correlate(xa, xa, mode = 'full')[-n:]
    #assert np.allclose(r, np.array([(xa[:n-k]*xa[-(n-k):]).sum() for k in range(n)]))
    #result = r/(variance*(np.arange(n, 0, -1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = r/(variance*n)
    return result


def absdispersion(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.std(x) ** 2. / abs(np.mean(x))
    return res

def tee(log_filename, *args):
    with open(log_filename, "a") as f:
       print(*args, file=f) 
    print(*args)


def interindiv_dist(eucl):
    intdist = np.array([e.sum() / (e.shape[0]**2) for e in eucl])
    return intdist

def mean_neighbors_dist(eucl, fop=85):
    res = []
    for e in eucl:
        mask = e<=fop
        res.append(e[mask].sum() / mask.sum())
    return np.array(res)


def min_fop_dist(eucl, fop=85):
    res = []
    for e in eucl:
        e2 = np.copy(e)
        mask = e2>fop
        e2[mask] = np.nan
        np.fill_diagonal(e2, np.nan)
        min_fop_ratios = np.nanmin(e2, 0) / fop
        res.append(min_fop_ratios)
    return np.array(res)



def min_max_dist(eucl, fop=85):
    res = []
    for e in eucl:
        e2 = np.copy(e)
        mask = e2>fop
        e2[mask] = np.nan
        np.fill_diagonal(e2, np.nan)
        min_max_ratios = np.nanmin(e2, 0) / np.nanmax(e2, 0)
        res.append(min_max_ratios)
    return np.array(res)


def local_occupancy(eucl, fop=85, agent_radius=17):
    res = []
    agent_surf = np.pi * (agent_radius**2.)
    fop_surf = np.pi * (fop**2.)
    for e in eucl:
        mask = e<=fop
        occ = 1. - (fop_surf - mask.sum(0) * agent_surf) / fop_surf
        res.append(occ)
    return np.array(res)


class KilomboLauncher:
    def __init__(self, exec_path = "../limmsswarm", base_dir = "."):
        self.exec_path = os.path.abspath(exec_path)
        self.base_dir = base_dir

    def create_config(self, config_path, config):
        with open(config_path, "w") as f:
            f.write(json.dumps(config))
            f.flush()
            os.fsync(f)

    def analyse_kilombo_results(self, data_path):
        with open(data_path, "r") as f:
            raw_data = f.read()
        data = json.loads(raw_data)
        nb_entries = len(data)
        self.max_ticks = data[-1]["ticks"]
        self.nb_bots = len(data[0]["bot_states"])

        def get_state(x, key):
            return [s["state"][key] if key in s["state"] else np.nan for s in x["bot_states"]]

        def get_state2(x, b, key):
            state = data[x]["bot_states"][b]["state"]
            return state[key] if key in state else np.nan

        def get_vals(x, key):
            return [s[key] for s in x["bot_states"]]

        # X,Y positions
        self.x_position = np.array([get_vals(data[x], "x_position") for x in range(nb_entries)])
        self.y_position = np.array([get_vals(data[x], "y_position") for x in range(nb_entries)])

        # Time series lambda
        self.val_lambda = np.array([np.mean(get_state(data[x], "lambda")) for x in range(nb_entries)] )
        self.autocorr_lambda = np.array(
                [ estimated_autocorrelation([get_state2(x, b, "lambda") for x in range(nb_entries)]) for b in range(self.nb_bots)]).T
        self.mean_autocorr_lambda = np.mean(self.autocorr_lambda, axis=1)

        # Time series avg_lambda
        self.avg_lambda = np.array([np.mean(get_state(data[x], "avg_lambda")) for x in range(nb_entries)])
        self.autocorr_avg_lambda = np.array(
                [ estimated_autocorrelation([get_state2(x, b, "avg_lambda") for x in range(nb_entries)]) for b in range(self.nb_bots)]).T
        self.mean_autocorr_avg_lambda = np.mean(self.autocorr_avg_lambda, axis=1)

        last_bot_states = data[-1]
        # Retrieve the last avg_lambda value of each agents
        self.last_avg_lambda = np.mean(get_state(last_bot_states, "avg_lambda"))
#        if self.last_avg_lambda < 1e-3:
#            self.last_avg_lambda = 0.
        self.last_lambda = np.mean(get_state(last_bot_states, "lambda"))
#        if self.last_lambda < 1e-3:
#            self.last_lambda = 0.

        # Retrieve the values of t
        self.val_t = np.array([get_state(data[x], "t") for x in range(nb_entries)])

        # Retrieve the values of s
        self.val_s = np.array([get_state(data[x], "s") for x in range(nb_entries)])
        self.sum_s = np.sum(self.val_s, axis=1)
        self.mean_s = np.mean(self.val_s, axis=1)
        self.min_s = np.min(self.val_s, axis=1)
        self.max_s = np.max(self.val_s, axis=1)

        # Time series last_mse
        #self.val_last_mse_0 = np.array([get_state(data[x], "last_mse_0") for x in range(nb_entries)])
        #self.val_last_mse_0[self.val_last_mse_0<=0] = np.nan
        #self.val_last_mse_1 = np.array([get_state(data[x], "last_mse_1") for x in range(nb_entries)])
        #self.val_last_mse_1[self.val_last_mse_1<=0] = np.nan
        #self.val_last_mse_2 = np.array([get_state(data[x], "last_mse_2") for x in range(nb_entries)])
        #self.val_last_mse_2[self.val_last_mse_2<=0] = np.nan
        self.val_last_mse_0 = np.array(get_state(last_bot_states, "last_mse_0"))
        self.val_last_mse_0[self.val_last_mse_0<=0] = np.nan
        self.val_last_mse_1 = np.array(get_state(last_bot_states, "last_mse_1"))
        self.val_last_mse_1[self.val_last_mse_1<=0] = np.nan
        self.val_last_mse_2 = np.array(get_state(last_bot_states, "last_mse_2"))
        self.val_last_mse_2[self.val_last_mse_2<=0] = np.nan

        # Retrieve the current behavior of the agents
        self.current_behavior = np.array([get_state(data[x], "current_behavior") for x in range(nb_entries)])
        self.mean_current_behavior = np.mean(self.current_behavior, axis=1)

        # Retrieve the number of neighbors
        try:
            self.nb_neighbors = np.array([get_state(data[x], "nb_neighbors") for x in range(nb_entries)])
            self.mean_nb_neighbors = np.mean(self.nb_neighbors, axis=1)
        except Exception as e:
            self.nb_neighbors = np.array([])
            self.mean_nb_neighbors = np.nan

        # Retrieve the number of valid diffusions
        self.diffusion_valid = np.array([get_state(data[x], "diffusion_valid1") for x in range(nb_entries)])
        self.sum_diffusion_valid = np.sum(self.diffusion_valid, axis=1)
        self.mean_diffusion_valid = np.mean(self.diffusion_valid, axis=1)


        res = {}
        res["x_position"] = self.x_position
        res["y_position"] = self.y_position
        res["last_lambda"] = self.last_lambda
        res["lambda"] = self.val_lambda
        res["mean_autocorr_lambda"] = self.mean_autocorr_lambda
        res["last_avg_lambda"] = self.last_avg_lambda
        res["avg_lambda"] = self.avg_lambda
        res["mean_autocorr_avg_lambda"] = self.mean_autocorr_avg_lambda
        res["t"] = self.val_t
        res["s"] = self.val_s
        res["sum_s"] = self.sum_s
        res["mean_s"] = self.mean_s
        res["min_s"] = self.min_s
        res["max_s"] = self.max_s
        res["diffusion_valid"] = self.diffusion_valid
        res["sum_diffusion_valid"] = self.sum_diffusion_valid
        res["mean_diffusion_valid"] = self.mean_diffusion_valid
        res["current_behavior"] = self.current_behavior
        res["mean_current_behavior"] = self.mean_current_behavior
        res["nb_neighbors"] = self.nb_neighbors
        res["mean_nb_neighbors"] = self.mean_nb_neighbors
        res['last_mse_0'] = self.val_last_mse_0
        res['last_mse_1'] = self.val_last_mse_1
        res['last_mse_2'] = self.val_last_mse_2

        return res


    def _run_kilombo(self, instance_base_path, config_path, botstates_path, log_path):
        if botstates_path is None:
            cmd = [self.exec_path, "-p", config_path]
        else:
            cmd = [self.exec_path, "-p", config_path, "-b", botstates_path]
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=instance_base_path)
        except Exception as e:
            warnings.warn(f"ERROR during Kilombo execution of command: '{str(cmd)}' with exception: '{str(e)}' and output: '{e.output}'", RuntimeWarning)
            traceback.print_exc()
            output = None
        try:
            with open(os.path.join(instance_base_path, log_path), "wb") as f:
                f.write(output)
        except Exception as e:
            warnings.warn("ERROR saving Kilombo logs with command: %s" % str(cmd), RuntimeWarning)
            traceback.print_exc()
        return output is not None

    def _clean_dir(self, instance_base_path):
        shutil.rmtree(pathlib.Path(instance_base_path))

    def launch(self, config, botstates_path, keep_tmp_files = False, keep_bugged_files = False):
        # Init file paths
        config_hash = hash(frozenset(config.items()))
        instance_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S') + "_" + str(abs(config_hash)) + "_" + str(config['randSeed'])
        instance_base_path = os.path.join(self.base_dir, instance_name)
        # Create instance directory
        pathlib.Path(instance_base_path).mkdir(parents=True, exist_ok=True)
        # Set kilombo files path
        config_path = "kilombo.json"
        log_path = "log.txt"
        data_path = os.path.join(instance_base_path, "data.json")

        # Create kilombo config file
        config['stateFileName'] = os.path.abspath(data_path)
        config['GUI'] = 0
        config['arenaFileName'] = os.path.abspath(config['arenaFileName'])

        self.create_config(os.path.join(instance_base_path, config_path), config)

        # Run kilombo and get result data
        exec_success = self._run_kilombo(instance_base_path, config_path, botstates_path, log_path)
        if not exec_success:
            if not keep_bugged_files and not keep_tmp_files:
                self._clean_dir(instance_base_path)
            return None

        # Analyse the results from kilombo
        res = self.analyse_kilombo_results(data_path)
        #print(f"# {instance_name}: {res}")

        # Delete temporary files
        if not keep_tmp_files:
            self._clean_dir(instance_base_path)
        return res


def launch_kilombo(base_config, arenaFileName, icFileName, seed, launcher, args):
    config = copy.deepcopy(base_config)
    config['randSeed'] = seed
    config['arenaFileName'] = arenaFileName
    botstates_path = None if icFileName is None else os.path.abspath(icFileName)
    instance_res = launcher.launch(config, botstates_path,
            keep_tmp_files = base_config.get('keep_tmp_files', False), keep_bugged_files = base_config.get('keep_bugged_files', False))
    return instance_res



def compute_stats_per_arena(data, config, output_path, log_filename, arenaFileName):
    all_stats = {}

    # lambda
    last_avg_lambda = np.array([r["last_avg_lambda"] for r in data])
    last_lambda = np.array([r["last_lambda"] for r in data])
    tee(log_filename, f"mean_avg_lambda {last_avg_lambda}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        abs_dispersion1 = np.std(last_avg_lambda)**2. / abs(np.mean(last_avg_lambda))
    tee(log_filename, f"mean of all 'last_avg_lambda': {np.mean(last_avg_lambda)} std: {np.std(last_avg_lambda)} abs_dispersion: {abs_dispersion1}")
    all_stats['last_avg_lambda'] = last_avg_lambda
    all_stats['mean_last_avg_lambda'] = np.mean(last_avg_lambda)
    all_stats['absdispersion_last_avg_lambda'] = absdispersion(last_avg_lambda)
    all_stats['last_lambda'] = last_lambda
    all_stats['mean_last_lambda'] = np.mean(last_lambda)
    all_stats['absdispersion_last_lambda'] = absdispersion(last_lambda)
    avg_lambda = np.array([r["avg_lambda"] for r in data])
    val_lambda = np.array([r["lambda"] for r in data])

    # Autocorrelations
    mean_autocorr_avg_lambda = np.array([r["mean_autocorr_avg_lambda"] for r in data])
    #tee(log_filename, f"mean_autocorr_avg_lambda: {mean_autocorr_avg_lambda}")
    #s = ", ".join([str(x) for x in mean_autocorr_avg_lambda[0]])
    #tee(log_filename, f"mean_autocorr_avg_lambda: {s}")
    all_stats['mean_autocorr_avg_lambda'] = mean_autocorr_avg_lambda
    mean_autocorr_lambda = np.array([r["mean_autocorr_lambda"] for r in data])
    all_stats['mean_autocorr_lambda'] = mean_autocorr_lambda

    # Stats related to positions of the robots
    all_xy = np.array([ np.stack([x['x_position'], x['y_position']], 2) for x in data ])
    if config.get('disable_dispersion_stats', False) == False:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_eucl = np.array([ [sklearn.metrics.pairwise.euclidean_distances(xy) for xy in a] for a in all_xy ])
            all_stats['interindiv_dist'] = np.array([interindiv_dist(e) for e in all_eucl])
            all_stats['mean_neighbors_dist'] = np.array([mean_neighbors_dist(e, config['commsRadius']) for e in all_eucl])
            all_stats['min_fop_dist'] = np.array([min_fop_dist(e, config['commsRadius']) for e in all_eucl])
            all_stats['min_max_dist'] = np.array([min_max_dist(e, config['commsRadius']) for e in all_eucl])
            all_stats['local_occupancy'] = np.array([local_occupancy(e, config['commsRadius'], 17) for e in all_eucl])

    # Tessellation-related stats
    all_stats['tess_stats'] = tessellation.compute_stats_graph(arenaFileName, all_xy[:,-1], config['commsRadius'], config['arenaNormalizedArea'])
    if config.get('disable_tessellation', False) == False:
        all_stats['voronoi_stats'] = tessellation.compute_stats_voronoi(all_xy[:,-1], config)
        all_stats['tess_ref_stats'] = tessellation.compute_stats_ref_graph(arenaFileName, config['commsRadius'], config['arenaNormalizedArea'], config['nBots'], 5, 1e4)

    all_stats['last_diffusion_valid'] = np.array([a['mean_diffusion_valid'][-1] for a in data])

    # Plot Avg lambda
    simulationTime = config['simulationTime']
    x = np.linspace(0, simulationTime, len(avg_lambda[0]))
    y_ = np.array(avg_lambda).T
    ymean = np.mean(y_, axis = 1)
    ymin = np.min(y_, axis = 1)
    ymax = np.max(y_, axis = 1)
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    #ax.plot(mean_autocorr_avg_lambda[0])
    plt.plot(x, ymean, 'k-')
    plt.fill_between(x, ymin, ymax)
    #sns.lineplot(x, y)
    plt.savefig(os.path.join(output_path, "avg_lambda.pdf"))

    # Plot Autocorrelations
    simulationTime = config['simulationTime']
    x = np.linspace(0, simulationTime, len(mean_autocorr_avg_lambda[0]))
    y_ = np.array(mean_autocorr_avg_lambda).T
    ymean = np.mean(y_, axis = 1)
    ymin = np.min(y_, axis = 1)
    ymax = np.max(y_, axis = 1)
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    #ax.plot(mean_autocorr_avg_lambda[0])
    plt.plot(x, ymean, 'k-')
    plt.fill_between(x, ymin, ymax)
    #sns.lineplot(x, y)
    plt.savefig(os.path.join(output_path, "autocorr-avg_lambda.pdf"))

    return all_stats

    #all_res = []
    #for i in range(args.nbRuns):
    #    config = copy.deepcopy(base_config)
    #    config['randSeed'] = seed
    #    instance_res = launcher.launch(config, keep_tmp_files = True)
    #    all_res.append(instance_res)
    #    seed += 1
    #print(all_res)


def compute_stats_all_arenas(data_per_arena, stats_per_arena, config, output_path):
    stats = {}

    # Compute class centroids.. using the means (because the problem is 1D). TODO: For more dimensions, use K-Means instead.
    centroids = []
    for k,v in stats_per_arena.items():
        centroids.append(np.mean(v['last_avg_lambda']))
        v['centroid'] = centroids[-1]

    # Assign a predicted cluster to each run
    for i,(k,v) in enumerate(stats_per_arena.items()):
        d = np.abs(v['last_avg_lambda'].reshape(-1, 1) - np.array(centroids).reshape(1, -1))
        v['cluster_pred'] = np.argmin(d, axis=1)
        v['cluster_true'] = np.full(d.shape[0], i)

    # Regroup all cluster arrays
    y_true = np.array([v['cluster_true'] for k,v in stats_per_arena.items()]).flatten()
    y_pred = np.array([v['cluster_pred'] for k,v in stats_per_arena.items()]).flatten()

    # Print classification report
    print(sklearn.metrics.classification_report(y_true, y_pred, target_names=stats_per_arena.keys()))

    # Compute classification metrics
    stats['accuracy'] = sklearn.metrics.accuracy_score(y_true, y_pred)
    stats['f1'] = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    stats['precision'] = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    stats['recall'] = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    # Compute confusion matrix
    stats['confusion'] = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='true')

    # Diffusion Validity
    try:
        stats['last_diffusion_valid'] = np.min([ np.mean(a['last_diffusion_valid']) for a in stats_per_arena.values()])
    except Exception as e:
        pass

    # Centroid stats
    stats['centroids_std'] = np.std([a['centroid'] for a in stats_per_arena.values()])
    stats['mean_absdispersion_last_avg_lambda'] = np.mean([a['absdispersion_last_avg_lambda'] for a in stats_per_arena.values()])

    # Dispersion stats
    if config.get('disable_dispersion_stats', False) == False:
        tot_expe_steps = list(stats_per_arena.items())[0][1]['interindiv_dist'].shape[1]
        nb_steps_before_dispersed = 5 if tot_expe_steps > 5 else 0 # XXX adjust depending on parameters
        interindiv_dist = np.array([x['interindiv_dist'][:, nb_steps_before_dispersed:].mean() for x in stats_per_arena.values()])
        stats['mean_interindiv_dist'] = interindiv_dist.mean()
        stats['std_interindiv_dist'] = interindiv_dist.std()
        neighbors_dist = np.array([x['mean_neighbors_dist'][:, nb_steps_before_dispersed:].mean() for x in stats_per_arena.values()])
        stats['mean_neighbors_dist'] = neighbors_dist.mean()
        stats['std_neighbors_dist'] = neighbors_dist.std()
        stats['mean_min_fop_dist'] = np.nanmean(np.array([np.nanmean(x['min_fop_dist'][:, nb_steps_before_dispersed:]) for x in stats_per_arena.values()]))
        stats['mean_std_last_min_fop_dist'] = np.nanmean(np.array([np.nanstd(x['min_fop_dist'][:,-1,:], 1) for x in stats_per_arena.values()]))
        stats['std_mean_min_fop_dist'] = np.nanstd(np.array([np.nanmean(x['min_fop_dist'][:,nb_steps_before_dispersed:]) for x in stats_per_arena.values()]))
        stats['mean_min_max_dist'] = np.nanmean(np.array([np.nanmean(x['min_max_dist'][:, nb_steps_before_dispersed:]) for x in stats_per_arena.values()]))
        stats['mean_std_last_min_max_dist'] = np.nanmean(np.array([np.nanstd(x['min_max_dist'][:,-1,:], 1) for x in stats_per_arena.values()]))
        stats['std_mean_min_max_dist'] = np.nanstd(np.array([np.nanmean(x['min_max_dist'][:,nb_steps_before_dispersed:]) for x in stats_per_arena.values()]))
        stats['mean_local_occupancy'] = np.array([x['local_occupancy'][:, nb_steps_before_dispersed:].mean() for x in stats_per_arena.values()]).mean()
        stats['mean_std_last_local_occupancy'] = np.array([x['local_occupancy'][:,-1,:].std(1) for x in stats_per_arena.values()]).mean()
        stats['std_mean_local_occupancy'] = np.array([x['local_occupancy'][:, nb_steps_before_dispersed:].mean() for x in stats_per_arena.values()]).std()


    # Tessellation-related stats
    try:
        stats['mean_nb_conn_comp'] = np.nanmean(np.array([np.nanmean(x['tess_stats']['nb_conn_comp']) for x in stats_per_arena.values()]))
        stats['mean_degrees'] = np.nanmean(np.array([np.nanmean(x['tess_stats']['degrees']) for x in stats_per_arena.values()]))
        if config.get('disable_tessellation', False) == False:
            stats['mean_nb_conn_comp_ref'] = np.nanmean(np.array([np.nanmean(x['tess_ref_stats']['nb_conn_comp']) for x in stats_per_arena.values()]))
            mean_ref_edge_dists = np.array([a['tess_ref_stats']['edge_dists'].mean() for a in stats_per_arena.values()])
            err_edge_dists = np.array([a['tess_stats']['edge_dists'] - mean_ref for a,mean_ref in zip(stats_per_arena.values(), mean_ref_edge_dists) ])
            stats['mse_edge_dists'] = (err_edge_dists**2).mean(1)
            stats['mean_mse_edge_dists'] = stats['mse_edge_dists'].mean()
            stats['ratio_alg_conn'] = np.array([np.nanmean(a['tess_ref_stats']['alg_conn_main_conn_comp']) / np.nanmean(a['tess_stats']['alg_conn_main_conn_comp']) for a in stats_per_arena.values()])
            stats['mean_ratio_alg_conn'] = stats['ratio_alg_conn'].mean()
    except Exception as e:
        pass

    # Return all stats
    return stats


def _find_centroids(poly_ic, config, i, output_path, arenaFileName):
    centroids_ic = tessellation.find_polygon_centroids(poly_ic, config['nBots'])
    bot_states = [{'ID': i, 'direction': 0.0, 'x_position': c[1], 'y_position': c[0]} for i, c in enumerate(centroids_ic)]
    json_dict = {'ticks': 0, 'bot_states': bot_states}
    ic_filename = os.path.join(output_path, "ic_" + os.path.basename(arenaFileName).replace(".csv", f"_{i}.json"))
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(ic_filename, "w") as f:
        json.dump(json_dict, f)
    return ic_filename, centroids_ic

def init_ic_files(config, nb_runs, arenaFileName, output_path, initial_conditions_filename):
    if initial_conditions_filename is None and config.get('uniform_initial_distribution', False):
        # Create an initial conditions files based on a uniform repartition of the agents
        poly_ic = tessellation.polygon_from_csv(arenaFileName)
        poly_ic = tessellation.rescale_poly(poly_ic, config['arenaNormalizedArea'])

        # Find centroids for each run, through tessellation
        params = [[poly_ic, config, i, output_path, arenaFileName] for i in range(nb_runs)]
        with multiprocessing.Pool() as p:
            res_c = p.starmap(_find_centroids, params)
        ic_filenames, all_centroids_ic = list(zip(*res_c))
        centroids_ic = all_centroids_ic[-1]

#        ic_filenames = []
#        for i in range(nb_runs):
#            centroids_ic = tessellation.find_polygon_centroids(poly_ic, config['nBots'])
#            bot_states = [{'ID': i, 'direction': 0.0, 'x_position': c[1], 'y_position': c[0]} for i, c in enumerate(centroids_ic)]
#            json_dict = {'ticks': 0, 'bot_states': bot_states}
#            ic_filename = os.path.join(output_path, "ic_" + os.path.basename(arenaFileName).replace(".csv", f"_{i}.json"))
#            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
#            with open(ic_filename, "w") as f:
#                json.dump(json_dict, f)
#            ic_filenames.append(ic_filename)

        fig, ax = plt.subplots(figsize=(5,5))
        tessellation.plot_polygon(ax, poly_ic, facecolor="lightgrey", edgecolor="black")
        ax.plot(centroids_ic[:, 0], centroids_ic[:, 1], 'ro')
        ax.invert_yaxis()
        path_fig = os.path.join(output_path, "ic_" + os.path.basename(arenaFileName).replace(".csv", ".png"))
        plt.savefig(path_fig)

    else:
        ic_filenames = [initial_conditions_filename] * nb_runs

    return ic_filenames



if __name__ == "__main__":
    from multiprocessing import Pool
    from plots import *

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--inputPath', type=str, default='test.yaml', help = "Path of the agents state file")
    parser.add_argument('-o', '--outputPath', type=str, default="results", help = "Path of the resulting csv file")
    parser.add_argument('-e', '--execPath', type=str, default="limmsswarm", help = "Path of the kilombo executable")
    parser.add_argument('-n', '--nbRuns', type=int, default=0, help = "Number of runs")
    parser.add_argument('-N', '--maxProcesses', type=int, default=0, help = "Max number of processes, or 0")
    parser.add_argument('-s', '--seed', type=int, default=42, help = "Random seed")
    parser.add_argument('--keep-tmp-files', dest="keep_tmp_files", action="store_true", help = "Keep temporary files")
    parser.add_argument('--keep-bugged-files', dest="keep_bugged_files", action="store_true", help = "Keep bugged files")
    parser.add_argument('--extended-logs', dest="extended_logs", action="store_true", help = "Outputs extra infos in results files and plot them")
    args = parser.parse_args()

    #multiprocessing.set_start_method('spawn')

    # Init base paths
    config_base_filename = os.path.splitext(os.path.basename(args.inputPath))[0]
    output_path = os.path.join(args.outputPath, config_base_filename)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # Create and empty out log file
    log_filename = os.path.join(output_path, "log.txt")
    with open(log_filename, "w") as f:
        pass

    # Load base config
    base_config = yaml.safe_load(open(args.inputPath))
    initial_seed = args.seed
    # Set arena file names
    arenaFileNames = base_config['arenaFileNames']
    del base_config['arenaFileNames']
    # Set nb_runs
    nb_runs = args.nbRuns if args.nbRuns > 0 else base_config.get('nb_runs', 1) 
    # Set initial conditions
    if "initialConditionsFileNames" in base_config:
        initialConditionsFileNames = base_config["initialConditionsFileNames"]
        del base_config['initialConditionsFileNames']
    else:
        initialConditionsFileNames = None

    # Check if we need to save extended data/stats into the logs and make plots of these stats
    extended_logs = args.extended_logs or base_config.get('extended_logs', False)
    base_config['extended_logs'] = extended_logs
    # Check if tmp and bugged files should be kept
    base_config['keep_tmp_files'] = args.keep_tmp_files or base_config.get('keep_tmp_files', False)
    base_config['keep_bugged_files'] = args.keep_bugged_files or base_config.get('keep_bugged_files', False)

    tee(log_filename, f"config_filename: {args.inputPath}")
    tee(log_filename, f"seed: {initial_seed}")
    tee(log_filename, f"arenaFileNames: {arenaFileNames}")
    tee(log_filename, f"initialConditionsFileNames: {initialConditionsFileNames}")
    tee(log_filename, f"nb_runs: {nb_runs}")
    tee(log_filename, f"maxProcesses: {args.maxProcesses}")

    pool_args = {} if args.maxProcesses <= 0 else {'processes': args.maxProcesses}
    seed = initial_seed
    data_per_arena = {}
    stats_per_arena = {}
    with Pool(**pool_args) as pool:
        for i, arenaFileName in enumerate(arenaFileNames):
            tee(log_filename, f"\n### {arenaFileName} ###")
            expe_name = os.path.splitext(os.path.basename(arenaFileName))[0]
            output_path_full = os.path.join(output_path, expe_name)
            launcher = KilomboLauncher(base_dir = output_path_full, exec_path=args.execPath)
            icFileNames = init_ic_files(base_config, nb_runs, arenaFileName, output_path_full, None if initialConditionsFileNames is None else initialConditionsFileNames[i])
            params = [(base_config, arenaFileName, icFileName, s, launcher, args) for s, icFileName in zip(range(seed, seed+nb_runs), icFileNames)]
            seed += nb_runs
            data_per_arena[expe_name] = pool.starmap(launch_kilombo, params)
            stats_per_arena[expe_name] = compute_stats_per_arena(data_per_arena[expe_name], base_config, output_path_full, log_filename, arenaFileName)

    stats = compute_stats_all_arenas(data_per_arena, stats_per_arena, base_config, output_path)
    tee(log_filename, f"\n##############\n")
    tee(log_filename, "Final stats: ", stats)

    # Remove extended data/infos/stats if wanted
    if not extended_logs:
        for expe_name in data_per_arena.keys():
            data_per_arena[expe_name] = [{'avg_lambda': r['avg_lambda'], 'last_mean_diffusion_valid': r['mean_diffusion_valid'][-1]} for r in data_per_arena[expe_name]]
        if base_config.get('disable_dispersion_stats', False) == False:
            for expe_name in stats_per_arena.keys():
                del stats_per_arena[expe_name]['interindiv_dist']
                del stats_per_arena[expe_name]['mean_neighbors_dist']
                del stats_per_arena[expe_name]['min_fop_dist']
                del stats_per_arena[expe_name]['min_max_dist']
                del stats_per_arena[expe_name]['local_occupancy']

    # Save all results in a pickle file
    results = {'data_per_arena': data_per_arena, 'stats_per_arena': stats_per_arena, 'stats': stats, 'base_config': base_config}
    with open(os.path.join(output_path, "data.p"), "wb") as f:
        pickle.dump(results, f)

    # Make all plots
    try:
        # Find all icons
        #colors_dict = {'triangle': '#bf7f3f', 'square': '#bfbf3f', 'disk': '#3fbfbf', 'annulus': '#bf3fbf', '2': '#3f3f7f', '4': '#3fbf3f', '8': '#7f3f7f'}
        arena_names = list(results['data_per_arena'].keys())
        #colors = [colors_dict[x] for x in arena_names]
        #arena_icons = [os.path.join("arenas", f"icon_{a}.png") for a in arena_names]
        colors, arena_icons = get_colors_and_icons(arena_names, colors_dict)

        confusion_mat_plot(base_config, results, output_path, colors=colors, arena_icons=arena_icons)
        all_plots_ts(base_config, results, output_path, colors=colors, arena_icons=arena_icons)
        all_plots(base_config, results, output_path)

    except Exception as e:
        warnings.warn("ERROR while creating plots.", RuntimeWarning)
        traceback.print_exc()

    #new_config = copy.deepcopy(base_config)
    #new_config['arenaFileName'] = arenaFileName
    #params = [(config, )]
    #data_per_arena = pool.star(_sim, range(initial_seed, initial_seed+nb_runs))



# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
