from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from multiworld.envs.mujoco.sawyer_xyz.door.sawyer_door_open_finnMAML import SawyerDoorOpen_finnMAMLEnv
from multiworld.core.flat_goal_env import FlatGoalEnv

from multiworld.core.wrapper_env import NormalizedBoxEnv


import tensorflow as tf

import pickle

import rllab.misc.logger as logger

import doodad as dd




def experiment(variant, saveDir):




    fast_learning_rate = variant['flr']
    
    fast_batch_size = variant['fbs']  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
    meta_batch_size = 20  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
    max_path_length = 150
    num_grad_updates = 1
    meta_step_size = variant['mbs']

 


    baseEnv = SawyerDoorOpen_finnMAMLEnv()


    env = TfEnv(NormalizedBoxEnv(baseEnv))
   
    policy = MAMLGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=fast_learning_rate,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=(100,100),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = MAMLTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=fast_batch_size, # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=meta_batch_size,
        num_grad_updates=num_grad_updates,
        n_itr=1000,
        use_maml=True,
        step_size=meta_step_size,
        plot=False,
    )

    logger.set_snapshot_dir(saveDir)
    logger.add_tabular_output(saveDir+'progress.csv')

    algo.train()




saveDir = '/root/code/maml_rl/data/'

args = dd.get_args()

experiment(args['variant'], saveDir = saveDir)