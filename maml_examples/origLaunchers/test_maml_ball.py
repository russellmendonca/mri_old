from maml_examples.point_env_randgoal import PointEnvRandGoal
from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from multiworld.envs.mujoco.pointMass.ball_resetArgs import BallEnv


import csv
import joblib
import numpy as np
import pickle
import tensorflow as tf


initial_params_file = '/home/russellm/maml_rl/data/local/ball/trpomaml1_fbs20_mbs20_flr_0.1metalr_0.01_step11/itr_100.pkl'

for goal in range(20):


    stub(globals())
    env = TfEnv(BallEnv(goal_idx = goal))
  
 
    baseline =  LinearFeatureBaseline(env_spec=env.spec)
    algo = VPG(
        env=env,
        policy=None,
        load_policy=initial_params_file,
        baseline=baseline,
        batch_size=10000,  # 2x
        max_path_length=100,
        n_itr=10,
        optimizer_args={'init_learning_rate': 0.1, 'tf_optimizer_args': {'learning_rate': 0.01}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer},
        reset_arg = goal,

    )


    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=4,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=4,
        exp_prefix='ballmetaTest-ValSet',
        exp_name='goal'+str(goal),
        #plot=True,
    )

