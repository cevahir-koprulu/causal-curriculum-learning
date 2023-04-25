# An Empirical Study of Causality in Curriculum Learning

This is the code repository for the semester project of Cevahir Koprulu in *ECE 381V: Causality and RL* course at UT Austin.

This codebase is built on the repository of _Curriculum Reinforcement Learning via Constrained Optimal Transport_ (CURROT) by Klink et al. (ICML, 2022).

Web sources for CURROT:

Source code: https://github.com/psclklnk/currot/tree/icml (ICML branch)

Paper: https://proceedings.mlr.press/v162/klink22a.html

The implementation of Unlock domain is provided by the authors of the paper *Generalizing Goal-Conditioned Reinforcement Learning with Variational Causal Reasoning* (GRADER) by Ding et al. (NeurIPS, 2022). We thank Wenhao Ding for sharing the Unlock domain related scripts with us.

Web sources for GRADER:

Source code: https://github.com/GilgameshD/GRADER

Paper: https://arxiv.org/abs/2207.09081

We run our codebase on Ubuntu 20.04.5 LTS with Python 3.8.15

## Installation

The required packages are provided in a requirements.txt file which can be installed via the following command;
```bash
pip install -r requirements.txt
```

## How to run
To run a single experiment (training + evaluation), *run.py* can be called as follows (you can put additional parameters):
```bash
python run.py --train --eval 0 --env unlock_1d_in --type self_paced --target_type wide --DIST_TYPE gaussian --seed 1 # SPDL
python run.py --train --eval 0 --env unlock_1d_in --type wasserstein --target_type wide --seed 1 # CURROT
python run.py --train --eval 0 --env unlock_1d_in --type default --target_type wide --seed 1 # Default
python run.py --train --eval 0 --env unlock_1d_in --type plr --target_type wide --seed 1 # PLR
python run.py --train --eval 0 --env unlock_1d_in --type goal_gan --target_type wide --seed 1 # GoalGAN
```
The results demonstrated in the project report can be run via *run_{environment_name}_experiments.py* by changing environment_name to one of the following:
- unlock_1d_in_wide
- unlock_1d_in_ood_wide
- unlock_1d_in_ood_c_wide


## Evaluation
Under *misc* directory, there are three scripts:
1) *plot_expected_performance.py*: We use this script to plot the progression of expected return during training.
2) *plot_expected_success.py*: We use this script to plot the progression of expected success rate during training.
3) *plot_unlock_hom_performance.py*: We run this script to obtain heatmaps illustrating the average success rate of trained policies in all possible goals (door positions) in Unlock domain.
4) *sample_eval_contexts.py*: We run this script to draw contexts from the target context distributions and record them to be used for evaluation of trained policies.