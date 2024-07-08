#!/usr/bin/env python3

import numpy as np
from io import StringIO
import os
import csv
import copy
import plots
import sklearn
from sklearn.cluster import KMeans
import sklearn.metrics
import warnings
import multiprocessing

#import geopandas as gpd
import shapely
import shapely.affinity
from shapely.geometry import Polygon, Point

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

import networkx as nx
import geopandas as gpd
from geovoronoi import voronoi_regions_from_coords
import shapely


# https://stackoverflow.com/questions/55522395/how-do-i-plot-shapely-polygons-and-objects-using-matplotlib
# Plots a Polygon to pyplot `ax`
def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def polygon_from_csv(filename):
    all_coords = []
    current_coords = []
    idx_poly = 0
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n": # New polygon
                #print("NEW POLYGON", line)
                if len(current_coords) > 0:
                    if idx_poly > 0: # Reverse order
                        current_coords.reverse()
                    all_coords.append(current_coords)
                    idx_poly += 1
                current_coords = []
                continue
            fs = StringIO(line)
            reader = csv.reader(fs, delimiter=",")
            c = next(reader)
            current_coords.append([float(a) for a in c])
    if len(current_coords) > 0:
        if idx_poly > 0: # Reverse order
            current_coords.reverse()
        all_coords.append(current_coords)

    if len(all_coords) > 1:
        poly = Polygon(all_coords[0], all_coords[1:])
    else:
        poly = Polygon(all_coords[0])
    return poly

def rescale_poly(poly, target_area):
    fact = np.sqrt(target_area / poly.area)
    poly2 = shapely.affinity.scale(poly, fact, fact)
    dx = - (poly2.bounds[2] - poly2.bounds[0]) / 2. - poly2.bounds[0]
    dy = - (poly2.bounds[3] - poly2.bounds[1]) / 2. - poly2.bounds[1]
    poly3 = shapely.affinity.translate(poly2, dx, dy)
    return poly3


def _find_polygon_centroids(poly, nb_centroids = 25, nb_samples=100000):
    exterior = np.array(poly.exterior.xy)
    samples = np.random.uniform(exterior.min(), exterior.max(), (nb_samples, 2))
    inside = np.array([Point(a[0], a[1]).within(poly) for a in samples])
    valid_samples = samples[inside]
    kmeans = KMeans(init="k-means++", n_clusters=nb_centroids, n_init=1, verbose=0)
    kmeans.fit(valid_samples)
    return kmeans.cluster_centers_

def find_polygon_centroids(poly, nb_centroids = 25, nb_samples=100000):
    if multiprocessing.current_process().name == 'MainProcess': # If we are in the main process.
        # We use sklearn KMeans implementation, which cannot be run in the main process due to an
        #   unsolved bug with OpenMP/multiprocessing interaction: https://github.com/scikit-learn/scikit-learn/issues/23823
        # So we launch a child process to compute the centroids
        with multiprocessing.Pool() as p:
            res = p.apply(_find_polygon_centroids, [poly, nb_centroids, nb_samples])
    else: # If we are not in the main process
        res = _find_polygon_centroids(poly, nb_centroids, nb_samples)
    return res


def create_graph(points, fop=85):
    all_eucl = sklearn.metrics.pairwise.euclidean_distances(points)
    g = nx.Graph()
    g.add_nodes_from(list(range(len(points))))

    for i in range(all_eucl.shape[0]):
        for j in range(all_eucl.shape[1]):
            if i != j and all_eucl[i, j] < fop:
                g.add_edge(i, j, weight = all_eucl[i, j])
    return g



def _stats_graph(g):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = [d for n,d in g.degree()]
        nb_conn_comp = len(list(nx.connected_components(g)))
        g_dists = np.array([g.get_edge_data(*e)['weight'] for e in g.edges])
        edge_dists = g_dists.mean()
        sg = [g.subgraph(c) for c in nx.connected_components(g)]
        try:
            #alg_conn_main_conn_comp = nx.algebraic_connectivity(sg[0])
            alg_conn_main_conn_comp = np.nan # May be bugged.. and too slow. So disable alg conn computation
        except Exception as e:
            alg_conn_main_conn_comp = np.nan
    return degrees, nb_conn_comp, edge_dists, alg_conn_main_conn_comp


def compute_stats_voronoi(all_xy, config):
    if config is None:
        x_domain = ( all_xy[..., 0].min(), all_xy[..., 0].max())
        y_domain = ( all_xy[..., 1].min(), all_xy[..., 1].max())
    else:
        display_width = config.get("displayWidth", 500)
        display_height = config.get("displayHeight", 500)
        x_domain = (- display_width / 2., display_width / 2)
        y_domain = (- display_height / 2., display_height / 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        area_shape = shapely.geometry.Polygon([ [x_domain[0], y_domain[0]], [x_domain[0], y_domain[1]], [x_domain[1], y_domain[1]], [x_domain[1], y_domain[0]] ])

        areas_voronoi = []
        for i, xy in enumerate(all_xy):
            region_polys, region_pts = voronoi_regions_from_coords(xy, area_shape)
            areas = np.array([p.area for p in region_polys.values()])
            areas_voronoi.append(areas)

    stats = {}
    stats['areas'] = np.array(areas_voronoi)
    return stats


def compute_stats_graph(arena_csv, all_xy, fop=85, surface=70000):
    degrees = []
    nb_conn_comp = []
    edge_dists = []
    alg_conn_main_conn_comp = []

    for i, xy in enumerate(all_xy):
        g = create_graph(xy, fop)
        res_stats = _stats_graph(g)
        degrees.append(res_stats[0])
        nb_conn_comp.append(res_stats[1])
        edge_dists.append(res_stats[2])
        alg_conn_main_conn_comp.append(res_stats[3])

    stats = {}
    stats['degrees'] = np.array(degrees)
    stats['nb_conn_comp'] = np.array(nb_conn_comp)
    stats['edge_dists'] = np.array(edge_dists)
    stats['alg_conn_main_conn_comp'] = np.array(alg_conn_main_conn_comp)
    return stats


def _compute_stats_ref_graph_gen_graph(poly, nb_agents, nb_samples, fop, seed):
    np.random.seed(seed)
    centroids = find_polygon_centroids(poly, nb_agents, int(nb_samples))
    g = create_graph(centroids, fop)
    res_stats = _stats_graph(g)
    return res_stats

def compute_stats_ref_graph(arena_csv, fop=85, surface=70000, nb_agents=25, trials=50, nb_samples=int(1e4)):
    degrees = []
    nb_conn_comp = []
    edge_dists = []
    alg_conn_main_conn_comp = []

    # Prepare polygon
    poly = polygon_from_csv(arena_csv)
    poly = rescale_poly(poly, surface)

    # Generate graphs
    seed0 = np.random.randint(2**30)
    params = [[poly, nb_agents, nb_samples, fop, seed0 + i] for i in range(trials)]
    with multiprocessing.Pool() as p:
        res_stats = p.starmap(_compute_stats_ref_graph_gen_graph, params)

    for r in res_stats:
        degrees.append(r[0])
        nb_conn_comp.append(r[1])
        edge_dists.append(r[2])
        alg_conn_main_conn_comp.append(r[3])

    stats = {}
    stats['degrees'] = np.array(degrees)
    stats['nb_conn_comp'] = np.array(nb_conn_comp)
    stats['edge_dists'] = np.array(edge_dists)
    stats['alg_conn_main_conn_comp'] = np.array(alg_conn_main_conn_comp)
    return stats


############### MAIN ############### {{{1
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--arenaCsv', type=str, default='arenas.csv', help = "Path of the arena csv file")
    parser.add_argument('-o', '--outputFile', type=str, default='/tmp/a.pdf', help = "Path of the output pdf file")
    args = parser.parse_args()

    stats_ref = compute_stats_ref_graph(args.arenaCsv, 80, 70000, 25, 24, 1e4)
    print(stats_ref)

    poly = polygon_from_csv(args.arenaCsv)
    poly = rescale_poly(poly, 70000)
    centroids = find_polygon_centroids(poly, 35)
    print(centroids)

    fig, ax = plt.subplots(figsize=(5,5))
    plot_polygon(ax, poly, facecolor="lightgrey", edgecolor="black")
    #ax.plot(valid_samples[:, 0], valid_samples[:, 1], 'ko', markersize=2)
    ax.plot(centroids[:, 0], centroids[:, 1], 'ro')
    plt.savefig(args.outputFile)


    exterior = np.array(poly.exterior.xy)
    samples = np.random.uniform(exterior.min(), exterior.max(), (1000, 2))
    inside = np.array([Point(a[0], a[1]).within(poly) for a in samples])
    valid_samples = samples[inside]
    #ex_points = np.array([valid_samples[:80]] * 10)
    ex_points = []
    for i in range(10):
        s = valid_samples.copy()
        np.random.shuffle(s)
        ex_points.append(s[:40])
    stats = compute_stats_graph(args.arenaCsv, ex_points, 85, 70000)
    stats.update(compute_stats_voronoi(all_xy, None))
    print(stats)



# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
