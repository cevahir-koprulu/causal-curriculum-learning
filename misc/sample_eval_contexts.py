import sys
sys.path.insert(0, '..')
import os
import math
import numpy as np
from deep_sprl.experiments.unlock_1d_in_experiment import Unlock1DInExperiment
from deep_sprl.experiments.unlock_1d_ood_experiment import Unlock1DOoDExperiment
from deep_sprl.experiments.unlock_1d_ood_c_experiment import Unlock1DOoDCExperiment
from deep_sprl.experiments.point_mass_2d_heavytailed_experiment import PointMass2DHeavyTailedExperiment
from deep_sprl.experiments.lunar_lander_2d_heavytailed_experiment import LunarLander2DHeavyTailedExperiment
from pathlib import Path


def sample_contexts(target_sampler, bounds, num_contexts):
    lower_bounds = bounds["lower_bounds"]
    upper_bounds = bounds["upper_bounds"]
    contexts = np.clip(target_sampler(n=num_contexts), lower_bounds, upper_bounds)
    return contexts


def sample_rare_contexts(target_sampler, num_contexts, p_min, p_max):
    target_mean = setting["target_mean"]
    target_covariance = setting["target_covariance"]
    lower_bounds = setting["lower_bounds"]
    upper_bounds = setting["upper_bounds"]
    contexts = []
    for c in range(num_contexts):
        inside = False
        while not inside:
            context = np.clip(np.random.multivariate_normal(mean=target_mean, cov=target_covariance, ),
                                lower_bounds, upper_bounds)
            a = context - target_mean
            p = np.exp(-0.5 * np.matmul(np.matmul(a.T, np.linalg.inv(target_covariance)), a))
            inside = (p_min <= p <= p_max)
        contexts.append(context)
    return np.array(contexts)

def sample_rare_uniform_contexts(target_sampler, num_contexts, p_min, p_max):
    target_mean = setting["target_mean"]
    target_covariance = setting["target_covariance"]
    lower_bounds = setting["lower_bounds"]
    upper_bounds = setting["upper_bounds"]
    contexts = []
    for c in range(num_contexts):
        inside = False
        while not inside:
            context = []
            for d in range(lower_bounds.shape[0]):
                context.append(np.random.uniform(lower_bounds[d], upper_bounds[d], 1))
            context = np.array(context)[0]
            a = context - target_mean
            p = np.exp(-0.5 * np.matmul(np.matmul(a.T, np.linalg.inv(target_covariance)), a))
            inside = (p_min <= p <= p_max)
        contexts.append(context)
    return np.array(contexts)

def sample_contexts_hom(bounds, num_per_axis):
    lower_bounds = bounds["lower_bounds"]
    upper_bounds = bounds["upper_bounds"]
    dim = lower_bounds.shape[0]
    if dim == 1:
        contexts = np.linspace(lower_bounds[0], upper_bounds[0], num=num_per_axis)
    elif dim == 2:
        x, y = np.meshgrid(np.linspace(lower_bounds[0], upper_bounds[0], num=num_per_axis),
                           np.linspace(lower_bounds[1], upper_bounds[1], num=num_per_axis))
        x_ = x.reshape(-1, 1)
        y_ = y.reshape(-1, 1)
        contexts = np.concatenate((x_, y_), axis=1)
    return contexts

def main():
    ##################################
    num_contexts = 100
    eval_context_dir = f"{Path(os.getcwd()).parent}/eval_contexts"
    target_type = "wide"
    env = f"unlock_1d_in_{target_type}"
    all_contexts = False
    rare_contexts = False
    rare_contexts_uniform = False
    all_contexts_hom = True
    num_per_axis = 8
    ##################################

    if not os.path.exists(eval_context_dir):
        os.makedirs(eval_context_dir)

    if env[:-len(target_type)-1] == "unlock_1d_in":
        exp = Unlock1DInExperiment(base_log_dir="logs", curriculum_name="self_paced", learner_name="ppo",
                                    parameters={"TARGET_TYPE": target_type}, seed=1, device="cpu")
    elif env[:-len(target_type)-1] == "unlock_1d_ood":
        exp = Unlock1DOoDExperiment(base_log_dir="logs", curriculum_name="self_paced", learner_name="ppo",
                                    parameters={"TARGET_TYPE": target_type}, seed=1, device="cpu")
    elif env[:-len(target_type)-1] == "unlock_1d_ood_c":
        exp = Unlock1DOoDCExperiment(base_log_dir="logs", curriculum_name="self_paced", learner_name="ppo",
                                    parameters={"TARGET_TYPE": target_type}, seed=1, device="cpu")
    elif env[:-len(target_type) - 1] == "point_mass_2d_heavytailed":
        exp = PointMass2DHeavyTailedExperiment(base_log_dir="logs", curriculum_name="self_paced", learner_name="ppo",
                                               parameters={"TARGET_TYPE": target_type}, seed=1, device="cpu")
    elif env[:-len(target_type) - 1] == "lunar_lander_2d_heavytailed":
        exp = LunarLander2DHeavyTailedExperiment(base_log_dir="logs", curriculum_name="self_paced",
                                                   learner_name="ppo", parameters={"TARGET_TYPE": target_type},
                                                   seed=1, device="cpu")
    else:
        raise ValueError("Invalid environment")

    bounds = {
        "lower_bounds": exp.LOWER_CONTEXT_BOUNDS,
        "upper_bounds": exp.UPPER_CONTEXT_BOUNDS,
    }

    if rare_contexts:
        contexts = sample_rare_contexts(target_sampler=exp.target_sampler,
                                        num_contexts=num_contexts,
                                        p_min=0., p_max=0.1)
        np.save(os.path.join(eval_context_dir,f"{env}_eval_rare_contexts"), contexts)

    if rare_contexts_uniform:
        contexts = sample_rare_uniform_contexts(target_sampler=exp.target_sampler,
                                                num_contexts=num_contexts,
                                                p_min=0., p_max=0.1)
        np.save(os.path.join(eval_context_dir,f"{env}_eval_rare_uniform_contexts"), contexts)

    if all_contexts:
        contexts = sample_contexts(target_sampler=exp.target_sampler,
                                   bounds=bounds,
                                   num_contexts=num_contexts,)
        print(contexts)
        np.save(os.path.join(eval_context_dir,f"{env}_eval_contexts"), contexts)

    if all_contexts_hom:
        contexts = sample_contexts_hom(bounds=bounds,
                                       num_per_axis=num_per_axis,)
        print(contexts)
        np.save(os.path.join(eval_context_dir,f"{env}_eval_hom_contexts"), contexts)


if __name__ == "__main__":
    main()