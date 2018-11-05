from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_pickPlace_finnMAML import SawyerPickPlace_finnMAMLEnv
from multiworld.core.flat_goal_env import FlatGoalEnv

from multiworld.core.wrapper_env import NormalizedBoxEnv

from sandbox.rocky.tf.algos.vpg import VPG
import tensorflow as tf

import pickle

import rllab.misc.logger as logger

import doodad as dd




def experiment(variant, saveDir):



    initial_params_file = variant['initial_params_file']
   
    goalIndex = variant['goalIndex']
   
    init_step_size = variant['init_step_size']

    baseEnv = SawyerPickPlace_finnMAMLEnv()
    env = TfEnv(NormalizedBoxEnv(baseEnv))
    baseline = LinearFeatureBaseline(env_spec=env.spec)


    algo = VPG(
            env=env,
            policy=None,
            load_policy=initial_params_file,
            baseline=baseline,
            batch_size=10000,  # 2x
            max_path_length=150,
            n_itr=10,
            reset_arg=goalIndex,
            optimizer_args={'init_learning_rate': init_step_size, 'tf_optimizer_args': {'learning_rate': 0.5*init_step_size}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
        )

    import os

    saveDir = variant['saveDir']

    if os.path.isdir(saveDir)==False:
        os.mkdir(saveDir)

    logger.set_snapshot_dir(saveDir)
    logger.add_tabular_output(saveDir+'progress.csv')

    algo.train()






saveDir = '/root/code/maml_rl/data/'

args = dd.get_args()

experiment(args['variant'], saveDir = saveDir)

   

   
    
