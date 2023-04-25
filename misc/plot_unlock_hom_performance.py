import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from pathlib import Path
from collections import defaultdict
sys.path.insert(1, os.path.join(sys.path[0], '..'))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_results(base_dir, iteration, seeds, grids, plot_success):
    expected = []
    contexts = []
    score_idx = -1 if plot_success else 1
    for seed in seeds:
        perf_file = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", "performance_hom.npy")
        if os.path.exists(perf_file):
            results = np.load(perf_file)
            for i_c in range(results.shape[0]):
                if len(expected) <= i_c:
                    expected.append([results[i_c:(i_c+1), score_idx][0]])
                    contexts.append(results[i_c, 2:-2])
                else:
                    expected[i_c].append(results[i_c:(i_c+1), score_idx][0])
        else:
            print(f"No evaluation data found: {perf_file}")
    expected_grids = np.zeros_like(grids[0])
    for i_c, c in enumerate(contexts):
        c_x = 0 if c.shape[0]==1 else np.where(grids[0][0,:]>=c[1])[0][0]
        c_y = np.where(grids[1][:,0]>=c[0])[0][0] 
        if c.shape[0]==1:
            expected_grids[c_y,c_x] = np.mean(expected[i_c])
        else:
            expected_grids[c_x,c_y] = np.mean(expected[i_c])
    return expected_grids

def plot_results(base_log_dir, iteration, seeds, plot_success, env, setting, algorithms, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    title = setting["title"]
    grids = setting["grids"]

    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        print(algorithm)

        base_dir = os.path.join(base_log_dir, env, algorithm, model)
        expected_grids = get_results(base_dir=base_dir,iteration=iteration, 
                                    seeds=seeds, grids=grids, plot_success=plot_success)

        fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
        im = ax.imshow(expected_grids, vmin=0.0, vmax=1.0)
        # Show all ticks and label them with the respective list entries
        ax.set_yticks(grids[1][:,0]-1, grids[1][:,0].astype(int))

        # Loop over data dimensions and create text annotations.
        for i in range(grids[0].shape[0]):
            for j in range(grids[0].shape[1]):
                text = ax.text(j, i, f"{expected_grids[i, j]:.2f}",
                            ha="center", va="center", color="w")
        ax.set_title(label)
        ax.set_ylabel("Door on the right")
        if grids[0].shape[1] !=1:
            ax.set_ylabel("Door on the right")
            ax.set_xticks(grids[0][0,:]-1, grids[0][0,:].astype(int))
        else:
            ax.set_ylabel("")
            ax.set_xticks([], [])
        figname = f"{cur_algo}"
        extension = "png"
        print(f"{Path(os.getcwd()).parent}\\figures\\{env}_{figname}{figname_extra}.{extension}")
        plt.savefig(f"{Path(os.getcwd()).parent}\\figures\\{env}_{figname}{figname_extra}.{extension}",
                    dpi=500, bbox_inches='tight')

def main():
    base_log_dir = f"{Path(os.getcwd()).parent}\\logs"
    seeds = [1, 2, 3, 4, 5]
    iteration = 295
    env = "unlock_1d_in_wide"
    # env = "unlock_1d_ood_wide"
    # env = "unlock_1d_ood_c_wide"
    plot_success = True
    figname_extra = "_expected_success_HEAT" if plot_success else "_expected_HEAT"

    algorithms = {
        "unlock_1d_ood_c_wide": {
            "DEF": {
                "algorithm": "default",
                "label": "Default",
                "model": "ppo",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "ppo_DELTA=0.05_DIST_TYPE=gaussian_INIT_VAR=None_KL_EPS=0.05",
            },
            "CURROT": {
                "algorithm": "wasserstein",
                "label": "CURROT",
                "model": "ppo_DELTA=0.05_METRIC_EPS=0.5",
            },
            "GoalGAN": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "ppo_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
            },
            "PLR": {
                "algorithm": "plr",
                "label": "PLR",
                "model": "ppo_PLR_BETA=0.45_PLR_REPLAY_RATE=0.85_PLR_RHO=0.15",
            },
        },
        "unlock_1d_ood_wide": {
            "DEF": {
                "algorithm": "default",
                "label": "Default",
                "model": "ppo",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "ppo_DELTA=0.05_DIST_TYPE=gaussian_INIT_VAR=None_KL_EPS=0.05",
            },
            "CURROT": {
                "algorithm": "wasserstein",
                "label": "CURROT",
                "model": "ppo_DELTA=0.05_METRIC_EPS=0.5",
            },
            "GoalGAN": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "ppo_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
            },
            "PLR": {
                "algorithm": "plr",
                "label": "PLR",
                "model": "ppo_PLR_BETA=0.45_PLR_REPLAY_RATE=0.85_PLR_RHO=0.15",
            },
        },
        "unlock_1d_in_wide": {
            "DEF": {
                "algorithm": "default",
                "label": "Default",
                "model": "ppo",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "ppo_DELTA=0.05_DIST_TYPE=gaussian_INIT_VAR=None_KL_EPS=0.05",
            },
            "CURROT": {
                "algorithm": "wasserstein",
                "label": "CURROT",
                "model": "ppo_DELTA=0.05_METRIC_EPS=0.5",
            },
            "GoalGAN": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "ppo_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
            },
            "PLR": {
                "algorithm": "plr",
                "label": "PLR",
                "model": "ppo_PLR_BETA=0.45_PLR_REPLAY_RATE=0.85_PLR_RHO=0.15",
            },
        },
    }

    settings = {
        "unlock_1d_ood_c_wide":
            {
                "grids": np.meshgrid(np.linspace(1., 8., 8),
                                     np.linspace(1., 8., 8),),
                "fontsize": 14,
                "figsize": (10, 5),
                "title": 'Expected discounted return',
            },
        "unlock_1d_ood_wide":
            {
                "grids": np.meshgrid(np.linspace(1., 1., 1),
                                     np.linspace(1., 8., 8),),
                "fontsize": 14,
                "figsize": (10, 5),
                "title": 'Expected discounted return',
            },
        "unlock_1d_in_wide":
            {
                "grids": np.meshgrid(np.linspace(1., 1., 1),
                                     np.linspace(1., 8., 8),),
                "fontsize": 14,
                "figsize": (10, 5),
                "title": 'Expected discounted return',
            },
    }

    plot_results(
        base_log_dir=base_log_dir,
        iteration=iteration,
        plot_success=plot_success,
        seeds=seeds,
        env=env,
        setting=settings[env],
        algorithms=algorithms[env],
        figname_extra=figname_extra,
        )


if __name__ == "__main__":
    main()
