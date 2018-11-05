from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from maml_examples.point_env_randgoal import PointEnvRandGoal
from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from multiworld.envs.mujoco.sawyer_xyz.pick.sawyer_pick_resetArgs import SawyerPickEnv

from multiworld.envs.mujoco.pointMass.ball_resetArgs import BallEnv
from multiworld.core.flat_goal_env import FlatGoalEnv



import tensorflow as tf



def experiment(variant):


    fast_learning_rate = variant['fast_learning_rate']
    
    fast_batch_size = variant['fast_batch_size']  
    meta_batch_size = variant['meta_batch_size']
    max_path_length = variant['max_path_length']
    num_grad_updates = variant['num_grad_updates']
    meta_step_size = variant['meta_step_size']

    

    env = TfEnv(BallEnv())


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

    algo.train()