import subprocess

filepath = r"./run.py"
arguments = [
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'self_paced', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '1'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'self_paced', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '2'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'self_paced', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '3'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'self_paced', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '4'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'self_paced', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '5'],

            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'self_paced_with_cem', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '1'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'self_paced_with_cem', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '2'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'self_paced_with_cem', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '3'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'self_paced_with_cem', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '4'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'self_paced_with_cem', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '5'],
            
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'default', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '1'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'default', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '2'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'default', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '3'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'default', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '4'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'default', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '5'],
            
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'default_with_cem', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '1'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'default_with_cem', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '2'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'default_with_cem', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '3'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'default_with_cem', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '4'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'default_with_cem', '--learner', 'ppo', '--DIST_TYPE', 'cauchy', '--TARGET_TYPE', 'wide', '--seed', '5'],

            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'wasserstein', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--DELTA', '-50.0', '--seed', '1'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'wasserstein', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--DELTA', '-50.0', '--seed', '2'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'wasserstein', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--DELTA', '-50.0', '--seed', '3'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'wasserstein', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--DELTA', '-50.0', '--seed', '4'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'wasserstein', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--DELTA', '-50.0', '--seed', '5'],

            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'goal_gan', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '1'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'goal_gan', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '2'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'goal_gan', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '3'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'goal_gan', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '4'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'goal_gan', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '5'],

            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'alp_gmm', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '1'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'alp_gmm', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '2'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'alp_gmm', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '3'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'alp_gmm', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '4'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'alp_gmm', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '5'],
            
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'acl', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '1'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'acl', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '2'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'acl', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '3'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'acl', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '4'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'acl', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '5'],
            
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'plr', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '1'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'plr', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '2'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'plr', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '3'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'plr', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '4'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'plr', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '5'],

            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'vds', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '1'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'vds', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '2'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'vds', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '3'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'vds', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '4'],
            ['--train', '--eval', '--eval_type', '0', '--env', 'lunar_lander_2d_heavytailed', '--type', 'vds', '--learner', 'ppo', '--TARGET_TYPE', 'wide', '--seed', '5'],
]

for args in arguments:
    subprocess.call(["python", filepath] + [arg for arg in args])