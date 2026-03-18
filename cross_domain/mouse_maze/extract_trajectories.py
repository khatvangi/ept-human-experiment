#!/usr/bin/env python3
"""
extract learning curves from rosenberg et al. mouse maze data.

uses home-run distance (monotonic path to exit) per bout as the
primary learning metric. this captures the "sudden insight" phenomenon
where mice abruptly shift from long, wandering returns to short,
efficient paths home.

also extracts path-to-water distance for rewarded mice as a second metric.

output: mouse_trajectories.csv with columns:
  mouse_id, bout_index, home_run_distance, time_in_maze_s, group
"""

import sys
import os
import pickle
import numpy as np
import csv

# add rosenberg code to path
CODE_DIR = '/tmp/rosenberg_maze/code'
sys.path.insert(0, CODE_DIR)

# need to set working directory so LoadTraj finds outdata/
os.chdir('/tmp/rosenberg_maze')

from MM_Maze_Utils import NewMaze
from MM_Traj_Utils import LoadTraj, FindPathsToExit, FindPathsToNode, TimeInMaze

# build maze structure
ma = NewMaze(6)

# mouse groups (from notebooks)
RewNames = ['B1', 'B2', 'B3', 'B4', 'C1', 'C3', 'C6', 'C7', 'C8', 'C9']
UnrewNamesSub = ['B5', 'B6', 'B7', 'D3', 'D4', 'D5', 'D7', 'D8', 'D9']
AllNames = RewNames + UnrewNamesSub

OUTPUT_CSV = '/storage/EPT/ept_human_experiment/cross_domain/mouse_maze/mouse_trajectories.csv'

rows = []

for nickname in AllNames:
    group = 'rewarded' if nickname in RewNames else 'unrewarded'
    tf = LoadTraj(nickname + '-tf')
    n_bouts = len(tf.no)

    # -- metric 1: home run distance per bout --
    # FindPathsToExit returns [bout_idx, starting_node, node_distance, abs_frame]
    hr = FindPathsToExit(tf, ma)

    # build a dict: bout -> home_run_distance
    hr_by_bout = {}
    for row in hr:
        bout_idx = int(row[0])
        dist = int(row[2])
        hr_by_bout[bout_idx] = dist

    # -- metric 2: path to water (node 116) for rewarded mice --
    water_by_bout = {}
    if nickname in RewNames:
        ptn = FindPathsToNode(116, tf, ma)
        if len(ptn) > 0:
            # there can be multiple water visits per bout; take the longest path
            for row in ptn:
                bout_idx = int(row[0])
                dist = int(row[2])
                if bout_idx not in water_by_bout or dist > water_by_bout[bout_idx]:
                    water_by_bout[bout_idx] = dist

    # -- compute time-in-maze at the start of each bout --
    for bout_idx in range(n_bouts):
        abs_frame_start = tf.fr[bout_idx, 0]
        try:
            t_maze_s = TimeInMaze(abs_frame_start, tf)
        except Exception:
            t_maze_s = float('nan')

        hr_dist = hr_by_bout.get(bout_idx, '')
        water_dist = water_by_bout.get(bout_idx, '')

        rows.append({
            'mouse_id': nickname,
            'bout_index': bout_idx,
            'home_run_distance': hr_dist,
            'water_path_distance': water_dist,
            'time_in_maze_s': round(t_maze_s, 2) if not np.isnan(t_maze_s) else '',
            'group': group,
        })

# write csv
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'mouse_id', 'bout_index', 'home_run_distance',
        'water_path_distance', 'time_in_maze_s', 'group'
    ])
    writer.writeheader()
    writer.writerows(rows)

# summary stats
print(f"wrote {len(rows)} rows to {OUTPUT_CSV}")
print(f"mice: {len(AllNames)}")
print(f"  rewarded: {len(RewNames)}, unrewarded: {len(UnrewNamesSub)}")

# per-mouse summary
for nickname in AllNames:
    mouse_rows = [r for r in rows if r['mouse_id'] == nickname]
    n_bouts = len(mouse_rows)
    n_hr = sum(1 for r in mouse_rows if r['home_run_distance'] != '')
    n_water = sum(1 for r in mouse_rows if r['water_path_distance'] != '')
    print(f"  {nickname}: {n_bouts} bouts, {n_hr} home runs, {n_water} water paths")
