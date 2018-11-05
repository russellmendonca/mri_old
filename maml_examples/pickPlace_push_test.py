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
from sandbox.rocky.tf.algos.vpg import VPG
import tensorflow as tf
import pickle
import rllab.misc.logger as logger
import doodad as dd
import numpy as np
import random

def experiment(variant):

    seed = variant['seed']
    tf.set_random_seed(seed) ; np.random.seed(seed) ; random.seed(seed)
    initial_params_file = variant['initial_params_file']
    goalIndex = variant['goalIndex']
    init_step_size = variant['init_step_size']

    regionSize = variant['regionSize']
   
    mode = variant['mode']

    if 'docker' in mode:
        taskFilePrefix = '/root/code'
    else:
        taskFilePrefix = '/home/russellm'
   
    if variant['valRegionSize'] !=None:
        valRegionSize = variant['valRegionSize']

        tasksFile = taskFilePrefix + '/multiworld/multiworld/envs/goals/pickPlace_'+ valRegionSize + '_val.pkl'

    else:
        tasksFile = taskFilePrefix + '/multiworld/multiworld/envs/goals/pickPlace_'+regionSize+ '.pkl'

      
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

    algo = VPG(
            env=env,
            policy=None,
            load_policy=initial_params_file,
            baseline=baseline,
            batch_size=7500,  # 2x
            max_path_length=150,
            n_itr=10,
            reset_arg=goalIndex,
            optimizer_args={'init_learning_rate': init_step_size, 'tf_optimizer_args': {'learning_rate': 0.1*init_step_size}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
        )
    import os
    saveDir = variant['saveDir']
    currPath = ''
    for _dir in saveDir.split('/'):
        currPath +=_dir + '/'
        if os.path.isdir(currPath)==False:
            os.mkdir(currPath)

    logger.set_snapshot_dir(saveDir)
    logger.add_tabular_output(saveDir+'progress.csv')
    algo.train()

#mode = 'docker'
mode = ''
if 'docker' in mode:
    saveDir = '/root/code/maml_rl/data/'
    args = dd.get_args()    
    experiment(args['variant'])
else:
    seed = 0 ; envType = 'PickPlace' ;  regionSize = '20X20' ;
    #valRegionSize ='30X30'
    valRegionSize = None 
    initFile = '/home/russellm/maml_rl/metaTrainedPolicies/'+envType+'_flr0-05_fbs50.pkl' ; init_step_size = 0.05 ; OUTPUT_DIR = '/home/russellm/maml_rl/data/' 

    for index in range(15, 20):
    #for index in [14]:

        tf.reset_default_graph()
        if valRegionSize : 
            expName = 'Val_'+valRegionSize+'/seed_'+str(seed)+'/Task_'+str(index)
        else:
            expName = 'Train_set/seed_'+str(seed)+'/Task_'+str(index)

        variant =  {'goalIndex':index, 'initial_params_file': initFile, 'init_step_size' : init_step_size, 'saveDir':OUTPUT_DIR+expName+'/', 'seed' : seed,
                    'envType' : envType , 'regionSize' : regionSize, 'mode': mode, 'valRegionSize': valRegionSize}

        experiment(variant)
   
    
