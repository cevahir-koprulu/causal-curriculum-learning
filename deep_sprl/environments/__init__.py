from gym.envs.registration import register

register(
    id='ContextualPointMass2D-v1',
    max_episode_steps=200,
    entry_point='deep_sprl.environments.point_mass.contextual_point_mass_2d:ContextualPointMass2D'
)

register(
    id='ContextualLunarLander-v1',
    max_episode_steps=1000,
    reward_threshold=200,
    entry_point='deep_sprl.environments.lunar_lander.contextual_lunar_lander:ContextualLunarLander'
)

register(
    id='ContextualLunarLander2D-v1',
    max_episode_steps=1000,
    reward_threshold=200,
    entry_point='deep_sprl.environments.lunar_lander.contextual_lunar_lander_2d:ContextualLunarLander2D'
)

register(
    id='ContextualUnlock1DIn-v1',
    max_episode_steps=100,
    reward_threshold=1,
    entry_point='deep_sprl.environments.unlock.contextual_unlock_1d_in:ContextualUnlock1DIn'
)

register(
    id='ContextualUnlock1DInMBRL-v1',
    max_episode_steps=100,
    reward_threshold=1,
    entry_point='deep_sprl.environments.unlock.contextual_unlock_1d_in_mbrl:ContextualUnlock1DInMBRL'
)

register(
    id='ContextualUnlock1DOoDTr-v1',
    max_episode_steps=100,
    reward_threshold=1,
    entry_point='deep_sprl.environments.unlock.contextual_unlock_1d_ood_tr:ContextualUnlock1DOoDTr'
)

register(
    id='ContextualUnlock1DOoDTe-v1',
    max_episode_steps=100,
    reward_threshold=1,
    entry_point='deep_sprl.environments.unlock.contextual_unlock_1d_ood_te:ContextualUnlock1DOoDTe'
)

register(
    id='ContextualUnlock1DOoDCTr-v1',
    max_episode_steps=150,
    reward_threshold=1,
    entry_point='deep_sprl.environments.unlock.contextual_unlock_1d_ood_c_tr:ContextualUnlock1DOoDCTr'
)

register(
    id='ContextualUnlock1DOoDCTe-v1',
    max_episode_steps=150,
    reward_threshold=1,
    entry_point='deep_sprl.environments.unlock.contextual_unlock_1d_ood_c_te:ContextualUnlock1DOoDCTe'
)