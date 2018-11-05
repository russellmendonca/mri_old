from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from rllab.envs.point_env_randgoal import PointEnvRandGoal
import tensorflow as tf
import numpy as np
import random
import pickle
import rllab.misc.logger as logger
import doodad as dd
from doodad.exp_utils import setup




def experiment(variant):


    seed = variant['seed'] ; n_parallel = variant['n_parallel'] ; log_dir = variant['log_dir']
    setup(seed, n_parallel, log_dir)




    fast_learning_rate = variant['flr']
    
    fast_batch_size = variant['fbs']  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
    meta_batch_size = 20  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
    max_path_length = 100
    num_grad_updates = 1
    meta_step_size = variant['mlr']

    env = TfEnv(normalize(PointEnvRandGoal()))

   
    policy = MAMLGaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        grad_step_size=fast_learning_rate,
        hidden_nonlinearity=tf.nn.relu,
        hidden_sizes=variant['hidden_sizes'],
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

    algo.train()




args = dd.get_args()
# args = {'seed':1, 'n_parallel':1, 'log_dir': '/home/russellm/maml_rl/data/pointMass' , 'flr': 1, 'mlr': 0.01, 'hidden_sizes': (100,100), 'fbs': 20}
# experiment(args)
experiment(args['variant'])