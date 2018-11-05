from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_pick_and_place import SawyerPickPlaceEnv
from multiworld.envs.mujoco.sawyer_xyz.push.sawyer_push import  SawyerPushEnv 
from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.core.finn_maml_env import FinnMamlEnv
from multiworld.core.wrapper_env import NormalizedBoxEnv
import tensorflow as tf
import numpy as np
import random
import pickle
import rllab.misc.logger as logger
import doodad as dd




def experiment(variant):


    seed = variant['seed']

    tf.set_random_seed(seed) ; np.random.seed(seed) ; random.seed(seed)




    fast_learning_rate = variant['flr']
    
    fast_batch_size = variant['fbs']  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
    meta_batch_size = 20  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
    max_path_length = 150
    num_grad_updates = 1
    meta_step_size = variant['mlr']


    regionSize = variant['regionSize']

    if regionSize == '20X20':

       
        tasksFile = '/root/code/multiworld/multiworld/envs/goals/pickPlace_20X20_6_8.pkl'

    else:
        assert regionSize == '60X30'

        tasksFile = '/root/code/multiworld/multiworld/envs/goals/pickPlace_60X30.pkl'

    tasks = pickle.load(open(tasksFile, 'rb'))


    envType = variant['envType']

    if envType == 'Push':

       
        baseEnv = SawyerPushEnv(tasks = tasks)
    else:
        assert(envType) == 'PickPlace'

        baseEnv = SawyerPickPlaceEnv( tasks = tasks)
    env = FinnMamlEnv(FlatGoalEnv(baseEnv, obs_keys=['state_observation', 'state_desired_goal']))



    env = TfEnv(NormalizedBoxEnv(env))
   
   
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = MAMLTRPO(
        env=env,
        policy=None,
        load_policy = variant['init_param_file'],
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

    import os

    saveDir = variant['saveDir']

    if os.path.isdir(saveDir)==False:
        os.mkdir(saveDir)

    logger.set_snapshot_dir(saveDir)
    logger.add_tabular_output(saveDir+'progress.csv')

    algo.train()




args = dd.get_args()


experiment(args['variant'])