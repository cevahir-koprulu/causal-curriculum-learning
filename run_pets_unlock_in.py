# from IPython import display
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf
from deep_sprl.experiments.unlock_1d_in_experiment import Unlock1DInExperiment
from deep_sprl.environments.unlock.contextual_unlock_1d_in_mbrl import ContextualUnlock1DInMBRL
import deep_sprl.environments.unlock.env.reward_fn as reward_fns
import deep_sprl.environments.unlock.env.termination_fn as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util

import gymnasium as gym
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# %load_ext autoreload
# %autoreload 2

mpl.rcParams.update({"font.size": 16})

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
trial_length = 100
num_trials = 1000
test_steps = 2500*5
ensemble_size = 5
log_dir = 'logs/unlock_1d_in_wide/pets/gaussian_mlp_cem/'

eval_path = f"{os.getcwd()}/eval_contexts/unlock_1d_in_wide_eval_contexts.npy"
eval_contexts = np.load(eval_path)

exp = Unlock1DInExperiment(base_log_dir="logs", curriculum_name="default", learner_name="ppo", 
                           parameters={"TARGET_TYPE": "wide"}, seed=1, device="cpu")

def run_pets_unlock_in_test(env, agent, trial_length, replay_buffer):
    num_context = eval_contexts.shape[0]
    test_rewards = []
    num_succ_eps_per_c = np.zeros((num_context, 1))
    for i in range(num_context):
        test_context = eval_contexts[i, :]
        env.context = test_context
        obs_test, _ = env.reset(None)    
        agent.reset()

        terminated_test = False
        total_reward_test = 0.0
        episode_step = 0
        # update_axes(axs, env.render(), ax_text, trial, steps_trial, all_rewards)
        while not terminated_test:
            next_obs_test, reward_test, terminated_test, truncated_test, _ = common_util.step_env_and_add_to_buffer(
                env, obs_test, agent, {}, replay_buffer)
            obs_test = next_obs_test
            total_reward_test += reward_test
            episode_step += 1
            if episode_step == trial_length:
                break

        num_succ_eps_per_c[i, 0] = 1.0*(total_reward_test>0)
        test_rewards.append(total_reward_test)
    test_rewards = np.array(test_rewards)
    return test_rewards, eval_contexts[:num_context, :], num_succ_eps_per_c

def run_pets_unlock_in(seed):
    seed_log_dir = f"{log_dir}/seed-{seed}"

    env = ContextualUnlock1DInMBRL()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # This functions allows the model to evaluate the true rewards given an observation 
    reward_fn = reward_fns.unlock
    # This function allows the model to know if an observation should make the episode end
    term_fn = termination_fns.unlock

    # Everything with "???" indicates an option with a missing value.
    # Our utility functions will fill in these details using the 
    # environment information
    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "_target_": "mbrl.models.GaussianMLP",
            "device": device,
            "num_layers": 3,
            "ensemble_size": ensemble_size,
            "hid_size": 128, # 200,
            "in_size": "???",
            "out_size": "???",
            "deterministic": False,
            "propagation_method": "fixed_model",
            # can also configure activation function for GaussianMLP
            "activation_fn_cfg": {
                "_target_": "torch.nn.LeakyReLU",
                "negative_slope": 0.01
            }
        },
        # options for training the dynamics model
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True,
        },
        # these are experiment specific options
        "overrides": {
            "trial_length": trial_length,
            "num_steps": num_trials * trial_length,
            "model_batch_size": 32,
            "validation_ratio": 0.05
        }
    }
    cfg = omegaconf.OmegaConf.create(cfg_dict)

    # Create a 1-D dynamics model for this environment
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    # Create a gym-like environment to encapsulate the model
    model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, generator=generator)

    replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)

    common_util.rollout_agent_trajectories(
        env,
        trial_length, # initial exploration steps
        planning.RandomAgent(env),
        {}, # keyword arguments to pass to agent.act()
        replay_buffer=replay_buffer,
        trial_length=trial_length
    )

    print("# samples stored", replay_buffer.num_stored)

    agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 5, #15,
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 5,
            "elite_ratio": 0.1,
            "population_size": 100, # 500,
            "alpha": 0.1,
            "device": device,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True,
            "clipped_normal": False
        }
    })

    agent = planning.create_trajectory_optim_agent_for_model(
        model_env,
        agent_cfg,
        num_particles=20
    )

    train_losses = []
    val_scores = []

    def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
        train_losses.append(tr_loss)
        val_scores.append(val_score.mean().item())   # this returns val score per ensemble model

    # Create a trainer for the model
    model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

    # Create visualization objects
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.75), gridspec_kw={"width_ratios": [1, 1]})
    ax_text = axs[0].text(300, 50, "")
        
    # Main PETS loop
    all_rewards = [0]
    all_test_rewards = []
    total_steps = 0

    for trial in range(num_trials):
        context = np.clip(exp.target_sampler(n=1), exp.LOWER_CONTEXT_BOUNDS, exp.UPPER_CONTEXT_BOUNDS)[0]
        env.context = context
        obs, _ = env.reset(None)    
        agent.reset()
        
        terminated = False
        total_reward = 0.0
        steps_trial = 0
        # update_axes(axs, env.render(), ax_text, trial, steps_trial, all_rewards)
        while not terminated:
            # --------------- Model Testing -----------------
            if total_steps % test_steps == 0:
                iter_log_dir = f"{seed_log_dir}/iteration-{int(total_steps // test_steps)}"
                os.makedirs(iter_log_dir, exist_ok=True)
                replay_buffer_tmp = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)
                test_rewards, eval_cs, successful_eps = run_pets_unlock_in_test(env=env, 
                                                                                agent=agent,
                                                                                trial_length=trial_length, 
                                                                                replay_buffer=replay_buffer_tmp)
                np.save(f"{iter_log_dir}/iterationtest_rewards.npy", test_rewards)
                np.save(f"{iter_log_dir}/eval_contexts.npy", eval_cs)
                np.save(f"{iter_log_dir}/successful_eps.npy", successful_eps)
                print(f"Iteration-{int(total_steps // test_steps)} - success: {100*np.mean(successful_eps)}%")
                all_test_rewards.append(test_rewards)

            # --------------- Model Training -----------------
            if steps_trial == 0:
                dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats
                
                dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                    replay_buffer,
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    ensemble_size=ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )
                    
                model_trainer.train(
                    dataset_train, 
                    dataset_val=dataset_val, 
                    num_epochs=50, 
                    patience=50, 
                    callback=train_callback,
                    silent=True)

            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, terminated, truncated, _ = common_util.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer)
                
            # update_axes(axs, env.render(), ax_text, trial, steps_trial, all_rewards)
            
            obs = next_obs
            total_reward += reward
            steps_trial += 1
            total_steps += 1
            
            if steps_trial == trial_length:
                break
        
        all_rewards.append(total_reward)
        print(f"Trial {trial} ended with reward={total_reward} at step={steps_trial}.")
      
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    ax[0].plot(train_losses)
    ax[0].set_xlabel("Total training epochs")
    ax[0].set_ylabel("Training loss (avg. NLL)")
    ax[1].plot(val_scores)
    ax[1].set_xlabel("Total training epochs")
    ax[1].set_ylabel("Validation score (avg. MSE)")
    # plt.show()
    plt.savefig("mbrl")

    return all_rewards, all_test_rewards, train_losses, val_scores

if __name__ == "__main__":
    num_seeds = 5

    all_train_rewards = []
    all_test_rewards = []
    all_train_losses = []
    all_val_scores = []
    for seed in range(1, num_seeds+1):
        rewards, test_rewards, train_losses, val_scores = run_pets_unlock_in(seed)
        all_train_rewards.append(rewards)
        all_test_rewards.append(test_rewards)
        all_train_losses.append(train_losses)
        all_val_scores.append(val_scores)
